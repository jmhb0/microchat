"""
python -m ipdb analysis_scripts/20241030_eval_after_v1_mcq_refiner_bot.py
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
from datetime import datetime
import glob
import csv

# results dir
sys.path.insert(0, "..")
sys.path.insert(0, ".")
from benchmark.build_raw_dataset.download_data import download_csv

verbose = 0

prompt_eval_templates = {
    0: {
        "about":
        "based on prompt from MMLU-pro https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/b7b9ffd84b2c21a5bfcf174fc65e5e6d74ca09a8/evaluate_from_api.py modified to add image",
        "template": """\
{{QUESTION}}
Think step by step and then output the answer in the format of \"The answer is (X)\" at the end." \

{{CHOICES}}
""",
        "regex_pattern": r"answer is \(?([0-9])\)?",
    }
}


def get_data(questions_source):
    if questions_source == 'jeffs_revised':
        url_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuDqK65cBcb0e5y-_DqK5HbFC3raPMP2isPBzTe8tg6vsfTl-7WkDI7NnKTzHJWQ/pub?gid=1746076346&single=true&output=csv"
        f_csv = dir_results_parent / "jeffs_choices.csv"
        if not f_csv.exists():
            download_csv(url_csv, f_csv)
        df = pd.read_csv(f_csv)
        do_shuffle = True

    elif questions_source == 'qkey3_ckey9':
        f_csv = "benchmark/data/formdata_0/question_strategy_3/df_questions_key_choices_9.csv"
        df = pd.read_csv(f_csv)
        shuffle = False  # already shuffle

    else:
        raise ValueError()

    return df, shuffle


def main(dir_rewrite, run_number, dir_results_parent, do_image_eval,
         do_language_only):
    key_prompt_eval = 0
    seed = 0
    model = "gpt-4o-2024-08-06"

    url_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuDqK65cBcb0e5y-_DqK5HbFC3raPMP2isPBzTe8tg6vsfTl-7WkDI7NnKTzHJWQ/pub?gid=1746076346&single=true&output=csv"
    f_csv = Path(dir_rewrite) / "jeffs_choices.csv"
    assert f_csv.exists()

    # source the right base questions 
    questions_source = 'jeffs_revised'
    questions_source = 'qkey3_ckey9'
    df, _ = get_data(questions_source)

    # get the rewritten questions from the bot run `dir_rewrite` (which depends on the run number)
    mcqs, mcqs_question, mcqs_choices, idxs_question = get_mcqs_from_bot(
        dir_rewrite)
    df_questions = df.loc[idxs_question]
    assert len(df_questions) == len(idxs_question)

    ## prepare all the prompt info
    cache_images, batch_prompts_imgs, batch_prompts_text, gts, idxs = get_prompts(
        df_questions, mcqs, key_prompt_eval)

    assert len(batch_prompts_text) == len(batch_prompts_imgs)
    assert len(batch_prompts_text) == len(idxs)

    seeds = [seed] * len(batch_prompts_text)
    # blind experiment change
    if key_prompt_eval == 1:
        batch_prompts_imgs = None

    print("calling gpt")
    responses = call_gpt_batch(texts=batch_prompts_text,
                               imgs=batch_prompts_imgs,
                               model=model,
                               json_mode=False,
                               seeds=seeds)
    cost = sum([c[1] for c in responses])
    msgs = [m[0] for m in responses]
    print(f"Cost of vlm call w choices${cost:.3f}")

    # regex out the predictions
    # preds_letter = []
    preds = _regex_predictions(
        msgs, prompt_eval_templates[key_prompt_eval]["regex_pattern"])

    # compute the basic stats
    gts = np.array(gts)
    preds = np.array(preds)
    acc = (gts == preds).sum() / len(gts)
    print(f"Accuracy {acc:.3f}")

    # get the old csv, and add accuracy results to it
    f_results_ = glob.glob(f"{dir_rewrite}/sum_run_{run_number:04d}*_sorted*")
    assert len(f_results_) == 1
    f_results = f_results_[0]
    df_results = pd.read_csv(f_results)
    df_results['question_key'] = [
        d[1] for d in df_results['log_str'].str.split("_")
    ]

    # make sure everything is lined up properly
    df_results.loc[idxs, 'correct'] = (gts == preds).astype(int)
    dir_results = dir_results_parent / f"res_{run_number}"
    dir_results.mkdir(exist_ok=True)
    f_save = dir_results / "results.csv"
    df_results.to_csv(f_save)

    df_questions['bot_question'] = mcqs_question
    df_questions['bot_choices'] = mcqs_choices
    df_questions['bot_code'] = df_results['code']
    df_questions['iterations'] = df_results['iterations']
    df_questions['cost'] = df_results['cost']

    ## also save the old questions.
    df_questions.loc[idxs, 'gt'] = -1
    df_questions.loc[idxs, 'gt'] = gts
    df_questions['pred'] = -1
    df_questions.loc[idxs, 'pred'] = preds

    df_questions.loc[idxs, 'correct'] = (preds == gts).astype(int)

    f_save_questions = dir_results / "df_questions.csv"
    df_questions.to_csv(f_save_questions)

    if 1:
        print(f"Per exit code accuracy ")
        print()

    if do_language_only:
        prompt_prefix = """The following question is supposed to be paired with an image. We will not provide the image, so answer to the best of your ability.\n\n"""
        batch_prompts_text_llm = [
            prompt_prefix + s for s in batch_prompts_text
        ]
        responses = call_gpt_batch(texts=batch_prompts_text_llm,
                                   imgs=None,
                                   model=model,
                                   json_mode=False,
                                   seeds=seeds)
        cost = sum([c[1] for c in responses])
        msgs = [m[0] for m in responses]
        print(f"Cost of vlm call w choices${cost:.3f}")

        # regex out the predictions
        # preds_letter = []
        preds_no_image = _regex_predictions(
            msgs, prompt_eval_templates[key_prompt_eval]["regex_pattern"])
        preds_no_image = np.array(preds_no_image)
        acc = (gts == preds_no_image).sum() / len(gts)
        print(f"Language only accuracy {acc:.3f}")

        df_questions['pred_no_image'] = -1
        df_questions.loc[idxs, 'pred_no_image'] = preds_no_image
        df_questions.loc[idxs, 'correct_no_image'] = (
            preds_no_image == gts).astype(int)

        f_save_questions = dir_results / "df_questions.csv"
        df_questions.to_csv(f_save_questions)

    ipdb.set_trace()
    pass


def get_mcqs_from_bot(dirs_rewrite):
    dirs = glob.glob(f"{dir_rewrite}/res_run_{run_number:04d}_*")
    assert len(dirs) == 1
    dirs_rewrite_qs = dirs[0]
    dirs_questions = glob.glob(f"{dirs_rewrite_qs}/question_*")
    idxs_question = [int(Path(d).stem.split("_")[1]) for d in dirs_questions]
    idxs_question = sorted(idxs_question)

    # get the question mcqs
    mcqs = []
    mcqs_question = []
    mcqs_choices = []
    for idx_question in idxs_question:
        files = glob.glob(
            f"{dirs_rewrite_qs}/question_{idx_question}/0_5_qa_iter*")
        nums = [int(Path(f).stem.split("_")[4]) for f in files]
        file = files[np.argmax(nums)]
        with open(file, 'r') as fp:
            mcq = json.load(fp)
        mcqs.append(mcq)

        mcqs_question.append(mcq['question_stem'])
        mcqs_choices.append({
            'chocices':
            dict(choices=mcq['choices'], correct_index=mcq['correct_index'])
        })
    return mcqs, mcqs_question, mcqs_choices, idxs_question


def get_prompts(df_questions, mcqs, key_prompt_eval):
    cache_images = {}
    batch_prompts_imgs = []
    batch_prompts_text = []
    gts = []
    idxs = []

    for i, (idx, row) in enumerate(df_questions.iterrows()):

        # get the images
        key_image = row['key_image']
        if key_image in cache_images.keys():
            imgs = cache_images[key_image]
        else:
            filenames = ast.literal_eval(row['fname_images'])
            try:
                imgs_pil = [Image.open(f).convert('RGB') for f in filenames]
                imgs = [np.array(img) for img in imgs_pil]
                cache_images[key_image] = imgs
                if verbose:
                    print(idx, [img.shape for img in imgs])

            except Exception as e:
                print(f"/nIssue with files {filenames}")
                print(e)
                continue
        batch_prompts_imgs.append(imgs)

        # recover the text question info
        mcq = mcqs[i]
        choices = mcq['choices']
        correct_index = mcq['correct_index']
        question_stem = mcq['question_stem']

        # construct the choices
        choices_str = ""
        for i, ch in enumerate(choices):
            choices_str += f"  ({i+1}): {ch}\n"

        # construct the text prompt
        prompt = prompt_eval_templates[key_prompt_eval]['template']
        prompt = prompt.replace("{{CHOICES}}", choices_str)
        prompt = prompt.replace("{{QUESTION}}", question_stem)
        batch_prompts_text.append(prompt)
        correct_index = mcq['correct_index']
        gts.append(correct_index)
        idxs.append(idx)

    return cache_images, batch_prompts_imgs, batch_prompts_text, gts, idxs


def _regex_predictions(texts: list[str], regex_pattern: str):
    preds = []
    for text in texts:
        pattern = regex_pattern
        match = re.search(pattern, text)
        if match is not None:
            pred = int(match.group(1)) - 1
            preds.append(pred)
        else:
            preds.append(-1)

    return preds


if __name__ == "__main__":
    run_number = 122
    do_image_eval = True
    do_language_only = True
    idxs_filter = True
    dir_rewrite = "analysis_scripts/results/20241028_v1_mcq_refiner_bot"

    dir_results_parent = Path(__file__).parent / "results" / Path(
        __file__).stem
    dir_results_parent.mkdir(exist_ok=True, parents=True)

    main(dir_rewrite, run_number, dir_results_parent, do_image_eval,
         do_language_only)

[
    'The sample undergoes self-repair mechanisms, maintaining image integrity.',
    'Electrons create new structural bonds, leading to clearer images.',
    'Electric fields nullify the high-energy impact, preserving quality.',
    'The electron absorption leads to compound crystallization, improving resolution.',
    'Prolonged electron contact causes changes that impair structural representation, reducing detail.'
]
