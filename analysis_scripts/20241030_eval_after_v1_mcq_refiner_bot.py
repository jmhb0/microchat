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
The following is a multiple choice question (with answers) and images. 

{{QUESTION}}
Think step by step and then output the answer in the format of \"The answer is (X)\" at the end." \

{{CHOICES}}
""",
        "regex_pattern": r"answer is \(?([0-9])\)?",
    }
}


def main(dir_rewrite, run_number, dir_results_parent):
    key_prompt_eval = 0
    seed = 0
    model = "gpt-4o-2024-08-06"

    url_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuDqK65cBcb0e5y-_DqK5HbFC3raPMP2isPBzTe8tg6vsfTl-7WkDI7NnKTzHJWQ/pub?gid=1746076346&single=true&output=csv"
    f_csv = Path(dir_rewrite) / "jeffs_choices.csv"
    assert f_csv.exists()
    df = pd.read_csv(f_csv)

    ## first get the rewritten questions

    # question folders
    dirs = glob.glob(f"{dir_rewrite}/res_run_{run_number:04d}_*")
    assert len(dirs) == 1
    dirs_rewrite_qs = dirs[0]
    dirs_questions = glob.glob(f"{dirs_rewrite_qs}/question_*")
    idxs_question = [int(Path(d).stem.split("_")[1]) for d in dirs_questions]
    idxs_question = sorted(idxs_question)
    df_questions = df.loc[idxs_question]
    assert len(df_questions) == len(idxs_question)

    # get the actual MCQ
    # class McqQA(BaseModel):
    #     question_stem: str
    #     choices: list[str]
    #     correct_index: int
    #     explanation: str

    mcqs = []
    for idx_question in idxs_question:
        files = glob.glob(
            f"{dirs_rewrite_qs}/question_{idx_question}/0_5_qa_iter*")
        nums = [int(Path(f).stem.split("_")[4]) for f in files]
        file = files[np.argmax(nums)]
        with open(file, 'r') as fp:
            mcq = json.load(fp)
        mcqs.append(mcq)

    ## prepare all the prompt info
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
    preds = []
    for msg in msgs:
        pattern = prompt_eval_templates[key_prompt_eval]["regex_pattern"]
        match = re.search(pattern, msg)
        if match is not None:
            pred = int(match.group(1)) - 1
            preds.append(pred)
        else:
            preds.append(-1)

    # compute the basic stats
    gts = np.array(gts)
    preds = np.array(preds)
    acc = (gts==preds).sum() / len(gts)
    print(acc)

    # get the old csv, and add accuracy results to it 
    f_results_ =  glob.glob(f"{dir_rewrite}/sum_run_{run_number:04d}*")
    assert len(f_results_) == 1 
    f_results = f_results_[0]
    df_results = pd.read_csv(f_results)
    df_results['question_key'] = [d[1] for d in df_results['log_str'].str.split("_")]

    # make sure everything is lined up properly 
    assert np.array_equal(df_results['question_key'].astype(int).values, np.array(idxs_question))
    df_results['correct'] = (gts==preds)

    dir_results = dir_results_parent / f"res_{run_number}"
    dir_results.mkdir(exist_ok=True)
    f_save = dir_results / "results.csv"
    df_results.to_csv(f_save)


if __name__ == "__main__":
    run_number = 1
    dir_rewrite = "analysis_scripts/results/20241028_v1_mcq_refiner_bot"

    dir_results_parent = Path(__file__).parent / "results" / Path(__file__).stem
    dir_results_parent.mkdir(exist_ok=True, parents=True)

    main(dir_rewrite, run_number, dir_results_parent)
