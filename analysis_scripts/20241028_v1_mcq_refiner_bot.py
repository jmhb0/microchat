"""
python -m ipdb analysis_scripts/20241028_v1_mcq_refiner_bot.py
"""

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

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

seed = 0  # controling the MCQ order
do_shuffle = True

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from benchmark.build_raw_dataset.download_data import download_csv
import prompts_20241028_v1_mcq_refiner_bot as prompts

idxs_question = [136, 137, 138, 139, 140, 142, 145]
idxs_question = [
    136, 137, 138, 139, 140, 142, 145, 176, 177, 178, 179, 180, 181, 187, 188,
    189, 190, 191, 192, 193, 194, 205, 206, 207, 538, 539, 540, 541, 542, 543
]
idxs_target = []

dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)
data_dir = Path("benchmark/data/formdata_0")
verbose = 0

models = [
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
]

model = "o1-preview-2024-09-12"
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
letters_to_idx = dict(zip(letters, range(len(letters))))
idx_to_letters = {v: k for k, v in letters_to_idx.items()}


def revise_mcq(cfg: OmegaConf,
               question_stem: str,
               choices: list[str],
               correct_index: int,
               dir_log: str,
               max_iters: int = 5,
               seed: int = 0):
    """
    Orhcestrator thread
    """
    _log_config(dir_log, cfg)

    question_stems_ = []
    choices_ = []
    evals_ = []  # memory of past answer attempts
    reflections_ = []  # memory of how past responses where successful
    rewrites_ = []
    check_rewrites_ = []
    costs = []

    # loop to improve things
    for iteration in range(max_iters):
        logging.info(f"Running iteration {iteration}")

        # log the current queastion and choices
        choices_.append(choices)
        question_stems_.append(question_stem)
        _log_qa(dir_log, iteration, question_stem, choices, correct_index)

        # evaluate current question without an image
        result_eval_mcq_noimage = evaluate_mcq_noimage(question_stem,
                                                       choices,
                                                       correct_index,
                                                       model=cfg.eval.model)
        evals_.append(result_eval_mcq_noimage)
        _log_eval(dir_log, iteration, result_eval_mcq_noimage, correct_index)

        # if eval is incorrect, then stop
        if not result_eval_mcq_noimage['is_correct']:
            logging.info(f"Successfully failed MCQ eval. Exiting")
            return question_stem, choices, result_eval_mcq_noimage
        if iteration == max_iters - 1:
            logging.info(f"Quitting after {max_iters} iterations")

        # reflect on how that was possible
        result_reflection = reflect_on_mcqnoimage_pass(
            conversation=result_eval_mcq_noimage['messages'],
            model=cfg.reflect.model,
            prompt_key=cfg.reflect.key)
        reflections_.append(result_reflection)
        _log_reflections(dir_log, iteration, result_reflection)

        # rewrite the question+distractors based on past reflections
        results_rewrite_qa = rewrite_qa(
            reflections=reflections_,
            prompt_key=cfg.rewrite.key,
            strucured_output_key=cfg.rewrite.strucured_output_key,
            model=cfg.rewrite.model,
            n_choices_target=cfg.rewrite.n_choices_target,
            question_stem=question_stem,
            choices=choices,
            correct_index=correct_index)
        rewrites_.append(results_rewrite_qa)
        _log_rewrites(dir_log, iteration, results_rewrite_qa)

        # check that the rewrite didn't change the meaning of the qa
        question_stem_new = results_rewrite_qa['mcq_qa_new']['question_stem']
        choices_new = results_rewrite_qa['mcq_qa_new']['choices']
        correct_index_new = results_rewrite_qa['mcq_qa_new']['correct_index']
        results_check_rewrite_issame = check_rewrite_issame(
            question_stem,
            choices[correct_index],
            question_stem_new,
            choices_new[correct_index_new],
            model=cfg.check_rewrite.model,
            key=cfg.check_rewrite.key,
            strucured_output_key=cfg.check_rewrite.strucured_output_key)
        check_rewrites_.append(results_check_rewrite_issame)
        _log_check_rewrite(dir_log, iteration, results_check_rewrite_issame)

        if not results_check_rewrite_issame['response']['is_equivalent']:
            # log and then return that it's changed, rather than try to fix it.
            raise ValueError(
                f"The rewrite prompt at iter {iteration} broke something")

        # update the current estimate
        question_stem = question_stem_new
        choices = choices_new
        correct_index = correct_index_new


def evaluate_mcq_noimage(question_stem,
                         choices,
                         correct_index,
                         model="o1-preview-2024-09-12",
                         key=0):
    """
    Run 
    """
    if key != 0:
        raise NotImplementedError()

    # "no image" prefix guidance + the standard CoT prompt + regex from MMLU-pro
    prompt_prefix = """The following question is supposed to be paired with an image. We will not provide the image, so answer to the best of your ability."""
    prompt_suffix = """Think step by step and then output the answer in the format of \"The answer is (X)\" at the end."""
    regex_pattern = r"answer is \(?([a-zA-Z])\)?"

    # make choices string
    choices_str = ""
    for letter, choice in zip(letters, choices):
        choices_str += f"({letter}) {choice}\n"

    # compose final prompt
    prompt_text = f"{prompt_prefix}\n{question_stem}\n{prompt_suffix}\n\n{choices_str}"

    # run gpt, extract prediction
    response = call_gpt(prompt_text, model=model, json_mode=False)
    response_text = response[0]
    pred_letter, pred_index = _extract_mc_answer(response_text, regex_pattern)
    is_correct = (correct_index == pred_index)

    cost = response[1]['cost'] if response[1] is not None else None

    return dict(messages=response[3],
                response_text=response_text,
                is_correct=is_correct,
                pred_index=pred_index,
                cost=cost)


def reflect_on_mcqnoimage_pass(conversation,
                               model='o1-preview-2024-09-12',
                               prompt_key=0):
    prompt_text = prompts.prompts_reflect[prompt_key]
    response = call_gpt(
        prompt_text,
        conversation=conversation,
        # overwrite_cache=True,
        json_mode=False)

    cost = response[1]['cost'] if response[1] is not None else None

    return dict(conversation=response[3], response_text=response[0], cost=cost)


def rewrite_qa(reflections: list[dict], prompt_key, model,
               strucured_output_key, n_choices_target, question_stem, choices,
               correct_index):
    """
    Each element of 'conversation' is a question + some

    strucured_output_key:
        0: structured output in prompt text. Then use llm to parse it. Used for o1 because they don't support it

    """
    conversations = [r['conversation'] for r in reflections]
    n_conversations = len(conversations)

    prompt = prompts.prompts_rewrite[prompt_key]
    prompt = prompt.replace("{{n_chat}}", str(n_conversations))
    prompt = prompt.replace("{{n_choices}}", str(n_choices_target))

    str_conversations = _stringify_conversations_lst(conversations)
    prompt = prompt.replace("{{conversations}}", str_conversations)

    # GPT call, enforcing the structured output optionally
    response_format = prompts.McqQA
    if strucured_output_key == 0:
        response_unstructured = call_gpt(prompt, model=model, json_mode=False)
        response = _enforce_llm_response_structure(response_unstructured,
                                                   response_format)
        cost = response_unstructured[1]['cost'] if response_unstructured[
            1] is not None else 0
        msg = response[0]
        messages = response_unstructured[0]

    elif strucured_output_key == 1:
        response = call_gpt(prompt,
                            model=model,
                            json_mode=False,
                            response_format=response_format)
        cost = response[1]['cost'] if response[1] is not None else 0
        msg = response[0]
        messages = response[3]

    else:
        raise NotImplementedError()

    logging.info(
        f"Cost ${cost:.2f} with model {model} for rewriting distractor")

    return dict(mcq_qa_new=msg, messages=messages)


def check_rewrite_issame(question_stem_1, answer_1, question_stem_2, answer_2,
                         key, model, strucured_output_key):
    """
    After revising question, run a check that the underlying content hasn't changed
    (Doing 1-indexing and not 0-indexing bc I thought the prompt might prefer it)
    """
    prompt = prompts.prompt_check_rewrite[key]
    prompt = prompt.replace("{{question_stem_1}}", question_stem_1)
    prompt = prompt.replace("{{answer_1}}", answer_1)
    prompt = prompt.replace("{{question_stem_2}}", question_stem_2)
    prompt = prompt.replace("{{answer_2}}", answer_2)

    response_format = prompts.PromptCheck
    if strucured_output_key == 0:
        response_unstructured = call_gpt(prompt, model=model, json_mode=False)
        response = _enforce_llm_response_structure(response_unstructured,
                                                   response_format)
        cost = response_unstructured[1]['cost'] if response_unstructured[
            1] is not None else 0
        msg = response[0]
        messages = response_unstructured[3]

    elif strucured_output_key == 1:
        response = call_gpt(prompt,
                            model=model,
                            json_mode=False,
                            response_format=response_format)
        cost = response[1]['cost'] if response[1] is not None else 0
        msg = response[0]
        messages = response[3]

    logging.info(
        f"Cost ${cost:.2f} with model {model} for checking distractor issame")

    return dict(response=msg, messages=messages)


def check_choices_have_no_simple_issues():
    # probably don't need this right now.
    raise NotImplementedError()


def _extract_mc_answer(text_response, regex_pattern):
    match = re.search(regex_pattern, text_response)

    if match is not None:
        pred_letter = match.group(1)
    else:
        pred_letter = 'None'

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
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
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
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
    np.random.seed(seed)
    idxs = np.arange(len(choices))
    idxs = np.random.permutation(idxs)
    choices_shuffled = [choices[idx] for idx in idxs]
    correct_index_new = int(np.where(idxs == correct_index)[0][0])

    return choices_shuffled, correct_index_new


def _enforce_llm_response_structure(response_unstructured,
                                    response_format: BaseModel,
                                    model: str = "gpt-4o-2024-08-06"):
    prompt = prompts.prompt_enforce_structure
    prompt = prompt.replace("{{original_response}}", response_unstructured[0])
    response = call_gpt(prompt, model=model, response_format=response_format)
    return response


def _log_config(dir_log, cfg):
    f_save = f"{dir_log}/cfg.json"
    with open(f_save, 'w') as fp:
        json.dump(OmegaConf.to_container(cfg), fp, indent=4)


def _log_qa(dir_log, iteration, question_stem, choices, correct_index):
    """ """
    f_save = dir_log / f"0_qa_iter_{iteration}.txt"
    str_log = _stringify_mcq_for_logging(question_stem, choices, correct_index)
    with open(f_save, "w") as fp:
        fp.write(str_log)


def _log_eval(dir_log, iteration, result_eval_mcq_noimage, correct_index):
    """
    By turning it into a string, we use newline, which makes logging more 
    readable.
    """
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
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
    str_log = _stringify_conversation_pretty(
        results_check_rewrite_issame['messages'])
    with open(dir_log / f"4_checkrewrite_iter_{iteration}.txt", 'w') as fp:
        fp.write(str_log)


def main():
    # config #
    model_o1 = "o1-preview-2024-09-12"
    model = model_o1
    model_o1mini = "o1-mini-2024-09-12"
    model_gpt4o = "gpt-4o-2024-08-06"

    n_choices_target = 5
    # yapf: disable
    cfg = dict(
        n_choices_target=5,
        eval=dict(model=model_o1, key=0),
        reflect=dict(model=model_o1, key=0),
        rewrite=dict(model=model_o1, key=0, strucured_output_key=0, n_choices_target=5),
        check_rewrite=dict(model=model_gpt4o, key=0, strucured_output_key=1),
    )
    cfg = dict(
        eval=dict(model=model_gpt4o, key=0),
        reflect=dict(model=model_gpt4o, key=0),
        rewrite=dict(model=model_gpt4o, key=0, strucured_output_key=1, n_choices_target=5),
        check_rewrite=dict(model=model_gpt4o, key=0, strucured_output_key=1),
    )
    # yapf: enable
    cfg = OmegaConf.create(cfg)
    idx_test = 207
    seed = 0
    do_shuffle = True
    seed_shuffle = idx_test + seed
    model = "o1-preview-2024-09-12"
    log_str = f"question_{idx_test}"
    # end #

    # collect the questions
    dir_log = dir_results / f"{log_str}_seed{seed}"
    dir_log.mkdir(exist_ok=True)
    url_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuDqK65cBcb0e5y-_DqK5HbFC3raPMP2isPBzTe8tg6vsfTl-7WkDI7NnKTzHJWQ/pub?gid=1746076346&single=true&output=csv"
    f_csv = dir_results / "jeffs_choices.csv"
    if not f_csv.exists():
        download_csv(url_csv, f_csv)
    df = pd.read_csv(f_csv)
    row = df.loc[idx_test]
    question_stem = row['revised_question']
    choices_jeff_fmt = row['multiple_choice']
    choices, correct_index = _process_choices_from_jeffs_sheet(
        choices_jeff_fmt)
    if do_shuffle:
        choices, correct_index = _shuffle_choices(choices, correct_index,
                                                  seed_shuffle)
    # str_log = _stringify_mcq_for_logging(question_stem, choices, correct_index)
    # _log_start(dir_log, question_stem, choices, correct_index)

    # run the revision bot
    revise_mcq(
        cfg,
        question_stem,
        choices,
        correct_index,
        dir_log,
        max_iters=5,
        seed=0,
    )


main()
