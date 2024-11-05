"""
python -m ipdb benchmark/refine_bot/bot.py
"""

from types import SimpleNamespace
import ipdb
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import logging
import ast
from PIL import Image
from models.openai_api import call_gpt_batch, call_gpt
import re
from pydantic import BaseModel
from omegaconf import OmegaConf
import logging
from datetime import datetime
import glob
import csv
import threading
import concurrent.futures

# results dir
sys.path.insert(0, "..")
sys.path.insert(0, ".")
from benchmark.refine_bot import prompts

file_lock = threading.Lock()

# maine revise loop thing
def revise_mcq(cfg: OmegaConf,
               question_stem: str,
               choices: list[str],
               correct_index: int,
               dir_log: str,
               max_iters: int = 5,
               seed: int = 0,
               log_str: str = ""):
    """
    main loop
    """
    # object to hold everything
    data = SimpleNamespace(question_stems_=[],
                           choices_=[],
                           evals_=[],
                           reflections_=[],
                           rewrites_=[],
                           check_rewrites_=[],
                           costs=[])

    # save the starting options
    question_stem_original = question_stem
    choices_original = choices
    correct_index_original = correct_index
    explanation = ""

    # logging dir
    Path(dir_log).mkdir(exist_ok=True)

    # loop 
    for iteration in range(max_iters):
        logging.info(f"[{log_str}] Running iteration {iteration}")

        # log the current queastion and choices
        data.choices_.append(choices)
        data.question_stems_.append(question_stem)
        _log_qa(dir_log, iteration, question_stem, choices, correct_index)

        # evaluate current question without an image
        result_eval_mcq_noimage = evaluate_mcq_noimage(question_stem,
                                                       choices,
                                                       correct_index,
                                                       seed=seed,
                                                       cfg_eval=cfg.eval)
        data.evals_.append(result_eval_mcq_noimage)
        _log_eval(dir_log, iteration, result_eval_mcq_noimage, correct_index)

        # if eval is incorrect, then stop
        if not result_eval_mcq_noimage['is_correct']:
            if iteration == 0:
                code = "SUCCESS_NO_CHANGE"
                logging.info(
                    f"[{log_str}] {code} MCQ already failed the image-free eval. Exiting"
                )
            else:
                code = "SUCCESS_REWRITE"
                logging.info(
                    f"[{log_str}] {code}  successfully failed MCQ eval after {iteration} iterations. Exiting"
                )
            return (code, iteration, question_stem, choices, data)

        # if max evals, then quit
        if iteration == max_iters - 1:
            code = "FAIL_ITERATIONS"
            logging.info(
                f"[{log_str}] {code} Quitting after {max_iters} iterations")
            return (code, iteration, question_stem, choices, data)

        # reflect on how that was possible
        result_reflection = reflect_on_mcqnoimage_pass(
            conversation=result_eval_mcq_noimage['messages'],
            cfg_reflect=cfg.reflect,
            seed=seed,
        )
        data.reflections_.append(result_reflection)
        _log_reflections(dir_log, iteration, result_reflection)

        # rewrite the question+distractors based on past reflections
        results_rewrite_qa = rewrite_qa(
            reflections=data.reflections_,
            cfg_rewrite=cfg.rewrite,
            question_stem_original=question_stem_original,
            choices_original=choices_original,
            correct_index_original=correct_index_original,
            seed=seed,
        )
        data.rewrites_.append(results_rewrite_qa)
        _log_rewrites(dir_log, iteration, results_rewrite_qa)

        # check that the rewrite didn't change the meaning of the qa
        question_stem_new = results_rewrite_qa['mcq_qa_new']['question_stem']
        choices_new = results_rewrite_qa['mcq_qa_new']['choices']
        correct_index_new = results_rewrite_qa['mcq_qa_new']['correct_index']
        explanation_new = results_rewrite_qa['mcq_qa_new']['explanation']

        results_check_rewrite_issame = check_rewrite_issame(
            question_stem_original,
            choices_original[correct_index_original],
            question_stem_new,
            choices_new[correct_index_new],
            seed=seed,
            cfg_check_rewrite=cfg.check_rewrite)
        data.check_rewrites_.append(results_check_rewrite_issame)
        _log_check_rewrite(dir_log, iteration, results_check_rewrite_issame)

        if not results_check_rewrite_issame['response']['is_equivalent']:
            # log and then return that it's changed, rather than try to fix it.
            code = "FAIL_REWRITE"
            logging.info(
                f"[{log_str}] {code} The rewrite prompt at iter {iteration} broke something. Exiting"
            )
            return (code, iteration, question_stem, choices, data)

        # update the current estimate
        question_stem = question_stem_new
        choices = choices_new
        correct_index = correct_index_new
        explanation = explanation_new


def evaluate_mcq_noimage(question_stem: str, choices: list[str],
                         correct_index: int, cfg_eval: OmegaConf, seed: int):
    """
    Run 
    """
    if cfg_eval.key not in (0, 1):
        raise NotImplementedError()

    # "no image" prefix guidance + the standard CoT prompt + regex from MMLU-pro
    prompt_prefix = """The following question is supposed to be paired with an image. We will not provide the image, so answer to the best of your ability."""
    prompt_suffix = """Think step by step and then output the answer in the format of \"The answer is (X)\" at the end."""
    regex_pattern = r"answer is \(?([a-zA-Z])\)?"

    # make choices string
    choices_str = ""
    letters = list("abcdefghijk")
    for letter, choice in zip(letters, choices):
        choices_str += f"({letter}) {choice}\n"

    # compose final prompt
    prompt_text = f"{prompt_prefix}\n{question_stem}\n{prompt_suffix}\n\n{choices_str}"

    # run gpt, extract prediction
    # ipdb.set_trace()
    response = call_gpt(prompt_text,
                        model=cfg_eval.model,
                        seed=seed,
                        verbose=False)
    response_text = response[0]
    pred_letter, pred_index = _extract_mc_answer(response_text, regex_pattern)
    is_correct = (correct_index == pred_index)

    cost = response[1]['cost'] if response[1] is not None else 0

    return dict(messages=response[3],
                response_text=response_text,
                is_correct=is_correct,
                pred_index=pred_index,
                cost=cost)


def reflect_on_mcqnoimage_pass(conversation: list[dict],
                               cfg_reflect: OmegaConf, seed: int):
    prompt_text = prompts.prompts_reflect[cfg_reflect.key]
    response = call_gpt(prompt_text,
                        model=cfg_reflect.model,
                        conversation=conversation,
                        seed=seed,
                        json_mode=False,
                        verbose=False)

    cost = response[1]['cost'] if response[1] is not None else 0
    return dict(conversation=response[3], response_text=response[0], cost=cost)


def rewrite_qa(reflections: list[dict], cfg_rewrite,
               question_stem_original: str, choices_original: list[str],
               correct_index_original: int, seed: int):
    # , prompt_key, model,
    #            strucured_output_key, n_choices_target, ):
    """
    Each element of 'conversation' is a question + some

    strucured_output_key:
        0: structured output in prompt text. Then use llm to parse it. Used for o1 because they don't support it

    """
    conversations = [r['conversation'] for r in reflections]
    n_conversations = len(conversations)

    prompt = prompts.prompts_rewrite[cfg_rewrite.key]
    prompt = prompt.replace("{{n_chat}}", str(n_conversations))
    prompt = prompt.replace("{{n_choices}}", str(cfg_rewrite.n_choices_target))

    prompt = prompt.replace("{{question_stem_original}}",
                            question_stem_original)
    prompt = prompt.replace("{{answer_original}}",
                            choices_original[correct_index_original])

    str_conversations = _stringify_conversations_lst(conversations)
    prompt = prompt.replace("{{conversations}}", str_conversations)

    # GPT call, enforcing the structured output optionally
    response_format = prompts.McqQA
    if cfg_rewrite.strucured_output_key == 0:
        response_unstructured = call_gpt(prompt,
                                         model=cfg_rewrite.model,
                                         seed=seed,
                                         verbose=False)
        response = _enforce_llm_response_structure(response_unstructured,
                                                   response_format, seed)
        cost = response_unstructured[1]['cost'] if response_unstructured[
            1] is not None else 0
        msg = response[0]  # structured version of the OG response.
        messages = response_unstructured[3]  # full convo from before

    elif cfg_rewrite.strucured_output_key == 1:
        response = call_gpt(prompt,
                            model=cfg_rewrite.model,
                            response_format=response_format,
                            seed=seed,
                            verbose=False)
        cost = response[1]['cost'] if response[1] is not None else 0
        msg = response[0]
        messages = response[3]

    else:
        raise NotImplementedError()

    return dict(mcq_qa_new=msg, messages=messages, cost=cost)


def check_rewrite_issame(question_stem_original: str, answer_original: str,
                         question_stem_new: str, answer_new: str, seed: str,
                         cfg_check_rewrite: OmegaConf):
    """
    After revising question, run a check that the underlying content hasn't changed
    (Doing 1-indexing and not 0-indexing bc I thought the prompt might prefer it)
    """
    prompt = prompts.prompt_check_rewrite[cfg_check_rewrite.key]
    prompt = prompt.replace("{{question_stem_1}}", question_stem_original)
    prompt = prompt.replace("{{answer_1}}", answer_original)
    prompt = prompt.replace("{{question_stem_2}}", question_stem_new)
    prompt = prompt.replace("{{answer_2}}", answer_new)

    response_format = prompts.PromptCheck
    if cfg_check_rewrite.strucured_output_key == 0:
        response_unstructured = call_gpt(prompt,
                                         model=cfg_check_rewrite.model,
                                         seed=seed,
                                         verbose=False)
        response = _enforce_llm_response_structure(response_unstructured,
                                                   response_format, seed)
        cost = response_unstructured[1]['cost'] if response_unstructured[
            1] is not None else 0
        msg = response[0]
        messages = response_unstructured[3]

    elif cfg_check_rewrite.strucured_output_key == 1:
        response = call_gpt(prompt,
                            model=cfg_check_rewrite.model,
                            seed=seed,
                            response_format=response_format,
                            verbose=False)
        cost = response[1]['cost'] if response[1] is not None else 0
        msg = response[0]
        messages = response[3]

    return dict(response=msg, messages=messages, cost=cost)


def check_choices_have_no_simple_issues():
    # probably don't need this right now.
    raise NotImplementedError()


def _extract_mc_answer(text_response, regex_pattern):
    matches = re.findall(regex_pattern, text_response)

    if len(matches) > 0:
        pred_letter = matches[-1]
    else:
        pred_letter = 'None'

    letters = list("abcdefghijk")
    letters_to_idx = dict(zip(letters, range(len(letters))))
    idx_to_letters = {v: k for k, v in letters_to_idx.items()}
    index = letters_to_idx.get(pred_letter, None)

    return pred_letter, index


def _stringify_mcq_for_logging(question_stem, choices, correct_index):
    """ 
    for logging, make the string where you put starts around the correct ans
          (a)   ... <wrong_answer> 
        **(b)** ... <right_answer>
          (c)   ... <wrong_answer>
    """
    str_log = question_stem + "\n"
    letters = list("abcdefghijk")
    letters_lookup = dict(zip(range(len(letters)), letters))
    letter = zip()
    for i, choice in enumerate(choices):
        letter = letters_lookup[i]
        if i == correct_index:
            str_log += f"**({letter})** {choice}\n"
        else:
            str_log += f"  ({letter})   {choice}\n"

    return str_log


def _stringify_conversations_lst(conversations):
    """
    Called by `rewrite_qa`, put the list of prior conversations into a string 
    that can be added to some llm prompt. 

    """
    str_convs = ""

    # iterate through multiple conversations
    for i in range(len(conversations)):
        str_convs += f"CONVERSATION {i+1}:\n "
        str_convs += _stringify_conversation(conversations[i])
        str_convs += "\n"

    return str_convs


def _stringify_conversation(conversation):
    """
    After processing, each 'conversation' will be represented as a list like this: 
    [ 
        {'role':'user', 'content': '....'},
        {'role':'assistant', 'content': '....'},
        {'role':'user', 'content': '....'},
        {'role':'assistant', 'content': '....'},
    ]
    However, in our implementation, the 'user' content is represented as
    {'role':'user', 'content': [{'type' : 'text', 'content' : '...'}]}

    This code also fixes that to the more standard form. There are assertions 
    to make sure thatn it is as we expect, so if the input 'conversations' 
    schema changes, then this code needs changing. 
    """
    # assert each convo turn is just text
    conv_lst = []
    for c in conversation:
        if c['role'] == 'user':
            assert len(c['content']) == 1
            conv_lst.append({
                "role": "user",
                "content": c['content'][0]['text']
            })
        elif c['role'] == 'assistant':
            assert type(c['content'])
            conv_lst.append(c)

        else:
            raise ValueError()
    str_conv = json.dumps(conv_lst, indent=2)
    return str_conv


def _stringify_conversation_pretty(conversation):
    """
    Pretty version of `_stringify_conversation` that shows the newlines to be 
    more readable for logging
    """
    # assert each convo turn is just text
    conv_str = ""
    for c in conversation:
        if c['role'] == 'user':
            assert len(c['content']) == 1
            conv_str += f"\n{80*'-'}\nUser\n{80*'-'}\n"
            conv_str += c['content'][0]['text']
        elif c['role'] == 'assistant':

            if type(c['content']) is str:
                conv_str += f"\n{80*'-'}\nAssistant\n{80*'-'}\n"
                conv_str += c['content']
            elif type(c['content']) is dict:
                conv_str += f"\n{80*'-'}\nAssistant (json response)\n{80*'-'}\n"
                conv_str += json.dumps(c['content'], indent=2)

        else:
            raise ValueError()

    return conv_str


def _process_choices_from_jeffs_sheet(choices: str):
    remove_strings = ["(Correct)", "(Incorrect)", "(correct)", "(incorrect)"]
    for s in remove_strings:
        choices = choices.replace(s, "")
    # make it a list
    choices_lst = choices.split("\n")
    assert len(choices_lst) in (4, 5)
    choices_lst = [c[3:].strip()
                   for c in choices_lst]  # remove the marker at the start

    return choices_lst, 0


def _shuffle_choices(choices, correct_index, seed_shuffle=0):
    np.random.seed(seed_shuffle)
    idxs = np.arange(len(choices))
    idxs = np.random.permutation(idxs)
    choices_shuffled = [choices[idx] for idx in idxs]
    correct_index_new = int(np.where(idxs == correct_index)[0][0])

    return choices_shuffled, correct_index_new


def _enforce_llm_response_structure(response_unstructured,
                                    response_format: BaseModel,
                                    seed: int,
                                    model: str = "gpt-4o-2024-08-06"):
    prompt = prompts.prompt_enforce_structure
    prompt = prompt.replace("{{original_response}}", response_unstructured[0])
    response = call_gpt(prompt,
                        model=model,
                        response_format=response_format,
                        seed=seed,
                        verbose=False)
    return response


def _log_qa(dir_log,
            iteration,
            question_stem,
            choices,
            correct_index,
            explanation=""):
    """ """
    # log as string
    f_save = dir_log / f"0_qa_iter_{iteration}.txt"
    str_log = _stringify_mcq_for_logging(question_stem, choices, correct_index)
    with open(f_save, "w") as fp:
        fp.write(str_log)

    # log as json for easier reading
    f_save = dir_log / f"0_5_qa_iter_{iteration}.json"
    mcq_object = prompts.McqQA(question_stem=question_stem,
                               choices=choices,
                               correct_index=correct_index,
                               explanation=explanation)
    with open(f_save, "w") as fp:
        json.dump(dict(mcq_object), fp, indent=2)


def _log_eval(dir_log, iteration, result_eval_mcq_noimage, correct_index):
    """
    By turning it into a string, we use newline, which makes logging more 
    readable.
    """
    letters = list("abcdefghijk")
    idx_to_letter = dict(zip(range(len(letters)), letters))

    msgs = result_eval_mcq_noimage['messages']
    assert len(msgs) == 2
    str_log = ""

    # prompt
    str_log += f"Prompt\n{80*'-'}\n"
    assert len(msgs[0]['content']) == 1
    str_log += msgs[0]['content'][0]['text']

    # response

    str_log += f"\n{80*'-'}\nResponse (target answer is {idx_to_letter[correct_index]})\n{80*'-'}\n"
    str_log += msgs[1]['content']

    with open(dir_log / f"1_eval_iter_{iteration}.txt", 'w') as fp:
        fp.write(str_log)


def _log_reflections(dir_log, iteration, result_reflection):
    str_log = _stringify_conversation_pretty(result_reflection['conversation'])
    with open(dir_log / f"2_reflection_iter_{iteration}.txt", 'w') as fp:
        fp.write(str_log)


def _log_rewrites(dir_log, iteration, results_check_rewrite_issame):
    str_log = _stringify_conversation_pretty(
        results_check_rewrite_issame['messages'])
    with open(dir_log / f"3_rewrite_iter_{iteration}.txt", 'w') as fp:
        fp.write(str_log)


def _log_check_rewrite(dir_log, iteration, results_check_rewrite_issame):
    messages = results_check_rewrite_issame['messages']
    str_log = ""
    # propmt
    str_log += f"Prompt\n{80*'*'}\n"
    str_log += messages[0]['content'][0]['text']

    # response
    res = results_check_rewrite_issame['response']
    str_log += f"\n{80*'*'}\n"
    str_log += f"Response, is_equivalent: {res['is_equivalent']}"
    str_log += f"\n{80*'*'}\n"
    str_log += f"is_equivalent: {res['explanation']}\n\n"

    with open(dir_log / f"4_checkrewrite_iter_{iteration}.txt", 'w') as fp:
        fp.write(str_log)


def _log_costs(cfg, evals_, reflections_, rewrites_, check_rewrites_,
               **kwargs):
    """
    For revising a single question, compute the final cost per stage. 
    """
    cost_eval = sum([c['cost'] for c in evals_])
    cost_reflect = sum([c['cost'] for c in reflections_])
    cost_rewrites = sum([c['cost'] for c in rewrites_])
    cost_check_rewrites = sum([c['cost'] for c in check_rewrites_])
    cost_total = cost_eval + cost_reflect + cost_rewrites + cost_check_rewrites

    # logging.info("Cost:")
    # logging.info(f"\t${cost_eval:.3f} on {cfg.eval.model} for eval")
    # logging.info(f"\t${cost_reflect:.3f} on {cfg.reflect.model} for reflect")
    # logging.info(f"\t${cost_rewrites:.3f} on {cfg.rewrite.model} for rewrite")
    # logging.info(
    #     f"\t${cost_check_rewrites:.3f} on {cfg.check_rewrite.model} for check rewrite"
    # )

    return cost_total


def _save_final_results(f_summary, log_str, cfg):
    """ 
    Final results saving. 
    In multiprocessing, the logging will be out of order, so reorder the rows.
    Also write some summary stats
    """
    df = pd.read_csv(f_summary)
    df['key_question'] = [int(n[1]) for n in df['log_str'].str.split("_")]
    df_ordered = df.sort_values("key_question")

    # reordering
    f_summary_ordered = Path(
        f_summary).parent / f"sum_{log_str}_samples_sorted.csv"
    df_ordered.to_csv(f_summary_ordered, index=False)

    # summarise
    str_log = ""
    str_log += f"API calls cost ${float(df['cost'].sum()):.2f}\n\n"
    str_log += str(df.groupby('code')['code'].count())
    str_log += "\n\n"
    str_log += str(df.groupby(['use_case', 'code'])['code'].count())
    str_log += "\n\n"
    str_log += str(df.groupby(['iterations'])['code'].count())
    str_log += "\n\n"
    str_log += str(df.groupby(['code', 'iterations'])['code'].count())
    str_log += "\n\n"
    str_log += json.dumps(OmegaConf.to_container(cfg), indent=4)
    f_summary_stats = Path(f_summary).parent / f"sum_{log_str}_stats.txt"
    with open(f_summary_stats, 'w') as fp:
        fp.write(str_log)

    logging.info(
        f"Final results in \n\t{f_summary_ordered}\n\t{f_summary_stats}")


def config_logger(dir_results_parent, cfg):
    # the run number is the highest run number in existing log files plus 1
    log_files = glob.glob(str(dir_results_parent / "run_*.log"))
    run_nums = [0] + [int(Path(f).stem.split("_")[1]) for f in log_files]
    run_num = max(run_nums) + 1

    # logging filename and folder name has run number and timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_str = f"run_{run_num:04d}_{timestamp}_{cfg.name}"
    log_filename = dir_results_parent / f"{log_str}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename),  # Logs everything to the file
            logging.StreamHandler(sys.stdout),  # Logs info and above to stdout
            logging.StreamHandler(
                sys.stderr)  # Logs errors and above to stderr
        ])
    logging.getLogger().handlers[2].setLevel(logging.ERROR)

    # experiment-level results directory
    dir_results = dir_results_parent / f"res_{log_str}"
    dir_results.mkdir(exist_ok=True)

    # save the config and dump to the logger
    f_save = dir_results / f"cfg.json"
    with open(f_save, 'w') as fp:
        json.dump(OmegaConf.to_container(cfg), fp, indent=4)
    logging.info("Config:")
    logging.info(json.dumps(OmegaConf.to_container(cfg), indent=4))

    # set up the results csv
    f_summary = dir_results_parent / f"sum_{log_str}_samples_async.csv"
    with file_lock:
        with open(f_summary, mode='a', newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                ["log_str", "iterations", "use_case", "code", "cost"])

    return dir_results, f_summary, log_str


def process_single_question(dir_log, cfg, question_stem, choices,
                            correct_index, f_summary, log_str, use_case):
    """
    Processes a single question with the MCQ refiner and do the basic logging
    """
    result = revise_mcq(
        cfg,
        question_stem,
        choices,
        correct_index,
        dir_log,
        max_iters=cfg.max_iters,
        seed=cfg.seed,
        log_str=log_str,
    )

    # unpack results
    return_code, iteration, question_stem, choices, data = result

    # evals_, reflections_, rewrites_, check_rewrites_
    cost_total = _log_costs(cfg, **vars(data))

    # save results
    with open(f_summary, mode='a', newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [log_str, iteration, use_case, return_code, f"{cost_total:.2f}"])

    return log_str, iteration, use_case, return_code, cost_total


def run(df: pd.DataFrame,
        cfg: OmegaConf,
        dir_results: str,
        do_multiprocessing: bool = False,
        do_shuffle: bool = False):
    """
    df must have the columns:
        question: values str
        choices: values are stringified dict with keys 
            'choices': list[str] 
            'correct_index': int
        use_case: int
    """
    # config #
    cfg = OmegaConf.create(cfg)
    dir_results, f_summary, log_str_main = config_logger(dir_results, cfg)

    # Prepare function args for all the questions
    func_args = []
    for idx_test, row in df.iterrows():
        log_str = f"question_{idx_test}"

        question_stem = row['question']
        choices_dict = ast.literal_eval(row['choices'])
        choices = choices_dict['choices']
        correct_index = choices_dict['correct_index']
        use_case = row['use_case']

        if do_shuffle:
            seed_shuffle = idx_test + cfg['seed']
            choices, correct_index = _shuffle_choices(choices, correct_index,
                                                      seed_shuffle)

        dir_log = dir_results / log_str

        # Collect arguments for this question
        args = (dir_log, cfg, question_stem, choices, correct_index, f_summary,
                log_str, use_case)
        func_args.append(args)

    # Run either in parallel or sequence based on flag
    if do_multiprocessing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
            futures = []
            for args in func_args:
                future = executor.submit(process_single_question, *args)
                futures.append(future)
            results = [future.result() for future in futures]

    else:
        results = []
        for args in func_args:
            result = process_single_question(*args)
            results.append(result)

    # Log summary of all results
    for log_str, iteration, use_case, return_code, cost_total in results:
        logging.info(
            f"Question {log_str}: {return_code} after {iteration} iterations. Cost: ${cost_total:.2f}"
        )

    _save_final_results(f_summary, log_str_main, cfg)


if __name__ == "__main__":

    model_o1 = "o1-preview-2024-09-12"
    model_o1mini = "o1-mini-2024-09-12"
    model_gpt4o = "gpt-4o-2024-08-06"
    model_gpt4omini = "gpt-4o-mini-2024-07-18"
    n_choices_target = 5

    # target questions
    idxs_question = list(range(20))

    # yapf: disable
    cfg = dict(
        name="standard",
        seed=0,
        max_iters=5,
        eval=dict(model=model_o1, key=0),
        reflect=dict(model=model_o1, key=0),
        rewrite=dict(model=model_o1, key=0, strucured_output_key=0, n_choices_target=n_choices_target),
        check_rewrite=dict(model=model_gpt4o, key=0, strucured_output_key=1),
    )
    # yapf: enable

    dir_results_parent =  Path("benchmark/refine_bot/results" )
    dir_results_parent.mkdir(exist_ok=True, parents=True)
    do_multiprocessing = False
    # do_multiprocessing = True

    # get the data
    questions_source = 'jeffs_revised'
    questions_source = 'qkey3_ckey9'
    df, do_shuffle = get_data(questions_source)
    df = df.iloc[:3]
    ipdb.set_trace()

    run(df,
        cfg,
        dir_results_parent,
        do_multiprocessing=do_multiprocessing,
        do_shuffle=False)
