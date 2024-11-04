"""
python -m ipdb benchmark/refine_bot/run_eval.py
"""
import ipdb
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import random
import json
from PIL import Image
import re
import pickle
import os
import glob

sys.path.insert(0, '.')
from models.openai_api import call_gpt_batch
from benchmark.refine_bot import run_experiments

dir_this_file = Path(__file__).parent

prompt_eval_templates = {
    0: {
        "about":
        "based on prompt from MMLU-pro https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/b7b9ffd84b2c21a5bfcf174fc65e5e6d74ca09a8/evaluate_from_api.py",
        "template": """\
The following is a multiple choice question (with answers). 
Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.


{{QUESTION}}

Options:
{{CHOICES}}
""",
        "regex_pattern": r"answer is \(?([0-9])\)?",
    },
    1: {
        "about": "based on prompt 0, but no images provided",
        "template": """\
The following is a multiple choice question (with answers).\
If an image is mentioned ignore this information and try your best to answer the question.
Think step by step and then output the answer in the format of \"The answer is (X)\" at the end."


{{QUESTION}}

Options:
{{CHOICES}}
""",
        "regex_pattern": r"answer is \(?([0-9])\)?",
    }
}


def eval_qa(df_questions,
            mcqs,
            key_prompt_eval,
            model='gpt-4o-mini',
            seed=0,
            verbose=False):
    """ 
    Run eval - both with and without the multi-choice options. 
    """
    assert len(df_questions) == len(mcqs)

    if 'gpt-4o' not in model:
        raise ValueError()

    batch_prompts_text = []
    batch_prompts_imgs = []
    idxs = []
    gts = []

    cache_images = {}
    if key_prompt_eval == 0:
        n_imgs = len(df_questions['key_image'].unique())
        print(f"Collecting {n_imgs} image sets")

    for i, (key_question, row) in enumerate(df_questions.iterrows()):

        # get the images
        if key_prompt_eval == 0:
            key_image = row['key_image']
            if key_image in cache_images.keys():
                imgs = cache_images[key_image]
            else:
                filenames = _get_filenames_from_key(row['key_image'])
                try:
                    imgs_pil = [
                        Image.open(f).convert('RGB') for f in filenames
                    ]
                    imgs = [np.array(img) for img in imgs_pil]
                    cache_images[key_image] = imgs
                    if verbose:
                        print(key_question, [img.shape for img in imgs])

                except Exception as e:
                    print(f"/nIssue with files {filenames}")
                    print(e)
                    continue
            batch_prompts_imgs.append(imgs)

        # construct the choices
        mcq = mcqs[i]
        choices = mcq['choices']
        choices_str = ""
        for i, ch in enumerate(choices):
            choices_str += f"  ({i+1}): {ch}\n"

        # construct the text prompt
        prompt = prompt_eval_templates[key_prompt_eval]['template']
        prompt = prompt.replace("{{CHOICES}}", choices_str)
        prompt = prompt.replace("{{QUESTION}}", mcq['question_stem'])
        batch_prompts_text.append(prompt)
        correct_index = mcq['correct_index']
        gts.append(correct_index)

        # save the indexes
        idxs.append(key_question)

    assert len(batch_prompts_text) == len(idxs)
    if key_prompt_eval == 0:
        assert len(batch_prompts_text) == len(batch_prompts_imgs)

    # a sense-check that the images are processed correctly
    if 0:
        batch_prompts_text = ["what is this image?"]
        batch_prompts_imgs = [batch_prompts_imgs[0]]
        responses = call_gpt_batch(texts=batch_prompts_text,
                                   imgs=batch_prompts_imgs,
                                   model=model,
                                   json_mode=False)
        msg = responses[0][0]

    # call gpt
    seeds = [seed] * len(batch_prompts_text)
    # blind experiment change
    if key_prompt_eval == 1:
        batch_prompts_imgs = None
    print(f"Running {model} on {len(batch_prompts_text)} samples")
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

    # save response
    gts = np.array(gts)
    preds = np.array(preds)

    return df_questions, msgs, preds, gts, cache_images


def get_refined_bot_mcqs(name, run):
    dir_rewrite = f"benchmark/refine_bot/results/run_experiments/{name}"
    if not Path(dir_rewrite):
        raise ValueError(f"no results folder for {dir_rewrite_parent}")

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

    # also recover the results csv
    # get the old csv, and add accuracy results to it
    f_results_ = glob.glob(f"{dir_rewrite}/sum_run_{run_number:04d}*_sorted*")
    assert len(f_results_) == 1
    f_results = f_results_[0]
    df_results = pd.read_csv(f_results)
    df_results['question_key'] = [
        d[1] for d in df_results['log_str'].str.split("_")
    ]

    return mcqs, mcqs_question, mcqs_choices, idxs_question, df_results


def _get_filenames_from_key(key, ):
    dir_ = f"benchmark/data/formdata_0/images/idx_{key:04d}"
    fs = sorted(
        [f for f in os.listdir(dir_) if os.path.isfile(os.path.join(dir_, f))])
    return [os.path.join(os.path.join(dir_, f)) for f in fs]


def select_mcqs_by_priority(dfs):
    """
    Select 'mcqs' values based on priority of 'code' values across multiple dataframes.
    
    Parameters:
    dfs (list): List of pandas DataFrames, each containing 'code' and 'mcqs' columns
    
    Returns:
    pandas.Series: Selected 'mcqs' values based on priority order
    """
    # Define priority order (highest to lowest)
    priority_order = [
        'SUCCESS_REWRITE',
        'SUCCESS_NO_CHANGE',
        'FAIL_ITERATIONS',
        'FAIL_REWRITE'
    ]
    
    # Create a dictionary to map codes to their priority (lower number = higher priority)
    priority_map = {code: i for i, code in enumerate(priority_order)}
    
    # Number of rows (assuming all dataframes have same number of rows)
    n_rows = len(dfs[0])
    
    # Initialize results with NaN
    selected_mcqs = pd.Series([np.nan] * n_rows)
    
    # For each row
    for row_idx in range(n_rows):
        best_priority = float('inf')
        best_mcq = np.nan
        
        # Check each dataframe
        for df in dfs:
            code = df.loc[row_idx, 'code']
            if code in priority_map:
                current_priority = priority_map[code]
                # If this code has higher priority (lower number) than what we've seen
                if current_priority < best_priority:
                    best_priority = current_priority
                    best_mcq = df.loc[row_idx, 'mcqs']
        
        selected_mcqs[row_idx] = best_mcq
    
    return selected_mcqs



if __name__ == "__main__":

    # config of whats in the other script
    df_results_lst = []

    seeds = [0,1,2,3,4]
    run_nums = [1,1,1,1,2]
    for (seed, run_number) in zip(seeds, run_nums):
        df, _, name = run_experiments.exp_1103_test150(seed=seed)
        mcqs, mcqs_question, mcqs_choices, idxs_question, df_results = get_refined_bot_mcqs(
            name, run_number)
        df_results['mcqs'] = mcqs
        df_results_lst.append(df_results)
        assert np.array_equal(df['key_question'].values, idxs_question)

    # ipdb.set_trace()
    # pass 
    df_choose_qs = select_mcqs_by_priority(df_results_lst)
    mcqs = df_choose_qs.values

    model = "gpt-4o-2024-08-06"
    do_language_only = True
    save_cached_images = False
    key_prompt_eval = 0

    df_questions, msgs, preds, gts, cache_images = eval_qa(
        df, mcqs, key_prompt_eval=key_prompt_eval, seed=0, model=model)
    acc = (gts == preds).sum() / len(gts)
    print(f"Acc VQA {acc:.4f} on {len(gts)} samples")

    df_questions['gt'] = gts
    df_questions['pred'] = preds
    df_questions['pred_correct'] = (preds == gts).astype(int)
    df_questions['pred_cot'] = msgs

    df = df_questions.merge(df_results, on="key_question", suffixes=('_q', '_r'))


    ipdb.set_trace()

    # if do_language_only:
    #     print(f"\n\nRunning language-only:")
    #     key_prompt_eval = 1
    #     _, msgs, preds, gts, _, _ = eval_qa(df,
    #                                         mcqs,
    #                                         key_prompt_eval=key_prompt_eval,
    #                                         seed=0,
    #                                         model=model)
    #     acc = (gts == preds).sum() / len(gts)
    #     print(f"Acc VQA, no image {acc:.1f}")

    #     df_questions['pred_no_img'] = preds
    #     df_questions['pred_no_img_correct'] = (preds == gts).astype(int)
    #     df_questions['pred_cot_no_img'] = msgs

    # print(f"Saving question-level eval to {f_eval_closed}")
    # df_questions.to_csv(f_eval_closed)

    # if save_cached_images:
    #     f_save_imgs = Path(f_eval_closed).parent / "images.pickle"
    #     print(f"\n\nSaving images to {f_save_imgs}")
    #     with open(f_save_imgs, "wb") as fp:
    #         pickle.dump(cache_images, fp)

    #     f_save_imgs
    #     print(f"Saving images to {f_eval_closed}")

    # ipdb.set_trace()
    # pass
