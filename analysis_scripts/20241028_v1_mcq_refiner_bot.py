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

model = "gpt-4o-2024-08-06"
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
               model: str = "o1-preview-2024-09-12",
               max_iters: int = 5,
               seed: int = 0):
    """
	Orhcestrator thread
	"""

    question_stems_ = []
    choices_ = []
    evals_ = []  # memory of past answer attempts
    reflections_ = []  # memory of how past responses where successful
    costs = []

    # loop to improve things
    for iteration in range(max_iters):
        choices_.append(choices)

        # evaluate current question without an image
        result_eval_mcq_noimage = evaluate_mcq_noimage(question_stem,
                                                       choices,
                                                       correct_index,
                                                       model=cfg.eval.model)
        evals_.append(result_eval_mcq_noimage)
        _log_eval(iteration, result_eval_mcq_noimage)

        # if eval is incorrect, then stop
        if not result_eval_mcq_noimage['is_correct']:
            return question_stem, choices, result_eval_mcq_noimage

        # reflect on how that was possible
        result_reflection = reflect_on_mcqnoimage_pass(
            conversation=result_eval_mcq_noimage['conversation'],
            model=cfg.reflect.model,
            prompt_key=cfg.reflect.key)
        reflections_.append(result_reflection)
        _log_reflections(iteration, result_reflection)

        result_rewrite_qa(reflections=reflections_, prompt_key=cfg.rewrite.key)
        ipdb.set_trace()

        if not is_similar:
            # log and then return that it's changed, rather than try to fix it.
            raise
            return


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
    pred_letter, pred_index = extract_mc_answer(response_text, regex_pattern)
    is_correct = (correct_index == pred_index)

    cost = response[1]['cost'] if response[1] is not None else None

    return dict(conversation=response[3],
                response_text=response_text,
                is_correct=is_correct,
                pred_index=pred_index,
                cost=cost)


def extract_mc_answer(text_response, regex_pattern):
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


def reflect_on_mcqnoimage_pass(conversation,
                               model='o1-preview-2024-09-12',
                               prompt_key=0):
    prompt_text = prompts.prompts_reflect[prompt_key]
    response = call_gpt(prompt_text,
                        conversation=conversation,
                        json_mode=False)

    cost = response[1]['cost'] if response[1] is not None else None

    return dict(conversation=response[3], response_text=response[0], cost=cost)


def result_rewrite_qa(reflections: list[dict], prompt_key):
    """
	Each element of 'conversation' is a question + some
	"""
    conversations = [r['conversation'] for r in reflections]
    n_conversations = len(conversations)

    prompt = prompts.prompts_rewrite[prompt_key]
    prompt = prompt.replace("{{n_chat}}", str(n_conversations))

    str_conversations = ""
    for i in range(len(conversations)):
        str_convs = f"CONVERSATION {i+1}:\n "
        str_convs += json.dumps(conversations[i], indent=2)
        str_convs += "\n"
    prompt = prompt.replace("{{conversations}}", str_convs)

    # response_format = prompts.McqQA

    ipdb.set_trace()
    response = call_gpt(prompt, model=model, json_mode=False)
    msg = response[0]
    cost = response[1]['cost'] if response[1] is not None else None

    ipdb.set_trace()

    pass


def check_question_answer_is_equivalent(question_stem_0,
                                        answer_0,
                                        question_stem_1,
                                        answer_1,
                                        model=model):
    """
	After revising question, run a check that the underlying content hasn't changed
	"""
    is_similar, = None  # run
    raise


def check_choices_have_no_simple_issues():
    # probably don't need this right now.
    pass


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


def _log_start(dir_log, question_stem, choices, correct_index):
    """ Assumes the correct answer is in (a) """
    f_save = dir_log / f"0_starting_question"
    str_log = _stringify_mcq_for_logging(question_stem, choices, correct_index)
    with open(f_save, "w") as fp:
        fp.write(str_log)


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


def _log_eval(iteration, result_eval_mcq_noimage):
    print("Warning have not implemented _log_eval")


def _log_reflections(iteration, result_reflection):
    print("Warning have not implemented _log_reflections")

def _log_rewrites(iteration, result_rewrite):
    print("Warning have not implemented _log_reflections")

def _remove_first_last_lines(text):
    lines = text.splitlines()  # Split text into lines
    return '\n'.join(lines[1:-1])  # Join all lines except the first and last



def main():
    # config #
    model = "o1-preview-2024-09-12"
    # model = "o1-mini-2024-09-12"
    cfg = dict(
        eval=dict(model=model, key=0),
        reflect=dict(model=model, key=0),
        rewrite=dict(model=model, key=0),
    )
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
    _log_start(dir_log, question_stem, choices, correct_index)

    # run the revision bot
    revise_mcq(
        cfg,
        question_stem,
        choices,
        correct_index,
        log_str,
        model=model,
        max_iters=5,
        seed=0,
    )


main()
