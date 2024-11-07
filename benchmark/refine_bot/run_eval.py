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

import hashlib
def checksum_md5(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()

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
        print(f"Running VQA with images")
        print(f"Collecting {n_imgs} image sets")
    else: 
        print(f"Running no-image QA")

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

    print("Lets check the size of the images ")
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


def get_refined_bot_mcqs(name, run_number):
    dir_rewrite = f"benchmark/refine_bot/results/run_experiments/{name}"
    print("*"*80)
    print(f"Getting results from directory {dir_rewrite}")
    
    if not Path(dir_rewrite):
        raise ValueError(f"no results folder for {dir_rewrite_parent}")

    # first recover the results csv
    # get the old csv, and add accuracy results to it
    f_results_ = glob.glob(f"{dir_rewrite}/sum_run_{run_number:04d}*_sorted*")
    f_results = f_results_[0]
    df_results = pd.read_csv(f_results)
    df_results['question_key'] = [
        d[1] for d in df_results['log_str'].str.split("_")
    ]
    # df_results.index = df_results["key_question"]

    dirs = glob.glob(f"{dir_rewrite}/res_run_{run_number:04d}_*")
    assert len(dirs) == 1
    dirs_rewrite_qs = dirs[0]
    dirs_questions = glob.glob(f"{dirs_rewrite_qs}/question_*")
    keys_question = [int(Path(d).stem.split("_")[1]) for d in dirs_questions]
    keys_question = sorted(keys_question)

    # get the question mcqs
    mcqs = []
    mcqs_question = []
    mcqs_choices = []
    for key_question in keys_question:
        row_ = df_results[df_results['key_question'] == key_question]
        assert len(row_) == 1
        row = row_.iloc[0]
        assert row['key_question'] == key_question

        code = row['code']
        if 'SUCCESS' not in code:
            file = f"{dirs_rewrite_qs}/question_{key_question}/0_5_qa_iter_0.json"
        else:
            files = glob.glob(
                f"{dirs_rewrite_qs}/question_{key_question}/0_5_qa_iter*")
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

    return mcqs, mcqs_question, mcqs_choices, keys_question, df_results


def _get_filenames_from_key(key, ):
    dir_ = f"benchmark/data/formdata_0/images/idx_{key:04d}"
    fs = sorted(
        [f for f in os.listdir(dir_) if os.path.isfile(os.path.join(dir_, f))])
    return [os.path.join(os.path.join(dir_, f)) for f in fs]


def select_mcqs_by_priority(dfs):
    """
    Select 'mcqs' values and corresponding DataFrame rows based on priority of 'code' values.

    Parameters:
    dfs (list): List of pandas DataFrames, each containing 'code' and 'mcqs' columns

    Returns:
    tuple: (selected_mcqs, selected_df_results) where:
        - selected_mcqs is a pandas.Series of selected MCQs
        - selected_df_results is a pandas DataFrame containing rows corresponding to selected MCQs
    """
    # Define priority order (highest to lowest)
    priority_order = [
        'SUCCESS_REWRITE', 'SUCCESS_NO_CHANGE', 'FAIL_ITERATIONS',
        'FAIL_REWRITE'
    ]

    # Create a dictionary to map codes to their priority (lower number = higher priority)
    priority_map = {code: i for i, code in enumerate(priority_order)}

    # Number of rows (assuming all dataframes have same number of rows)
    n_rows = len(dfs[0])

    # Initialize results with NaN
    selected_mcqs = pd.Series([np.nan] * n_rows)
    
    # Initialize list to keep track of which df and row to select
    selected_df_indices = []  # Will store tuples of (df_index, row_index)

    # For each row
    for row_idx in range(n_rows):
        best_priority = float('inf')
        best_mcq = np.nan
        best_df_idx = None

        # Check each dataframe
        for df_idx, df in enumerate(dfs):
            code = df.loc[row_idx, 'code']
            if code in priority_map:
                current_priority = priority_map[code]
                # If this code has higher priority (lower number) than what we've seen
                if current_priority < best_priority:
                    best_priority = current_priority
                    best_mcq = df.loc[row_idx, 'mcqs']
                    best_df_idx = df_idx

        selected_mcqs[row_idx] = best_mcq
        selected_df_indices.append((best_df_idx, row_idx))

    # Construct the selected results DataFrame
    selected_df_results = pd.DataFrame()
    for row_idx, (df_idx, orig_row_idx) in enumerate(selected_df_indices):
        if df_idx is not None:  # Only append if we found a valid selection
            selected_df_results = pd.concat([
                selected_df_results,
                pd.DataFrame([dfs[df_idx].iloc[orig_row_idx]])
            ])

    return selected_mcqs, selected_df_results


def exp_1103_test150_best_5():
    name = "mcqs_1104_best_5"
    seeds = [0, 1, 2, 3, 4]
    run_nums = [1, 1, 1, 1, 2]
    for (seed, run_number) in zip(seeds, run_nums):
        df, _, name = run_experiments.exp_1103_test150(seed=seed)
        mcqs, mcqs_question, mcqs_choices, keys_question, df_results = get_refined_bot_mcqs(
            name, run_number)
        df_results['mcqs'] = mcqs
        df_results_lst.append(df_results)
        assert np.array_equal(df['key_question'].values, keys_question)

    df_choose_qs, df_results = select_mcqs_by_priority(df_results_lst)
    mcqs = df_choose_qs.values

    return df, df_results, mcqs, name, run_number

def exp_1103_test150_o1mini_best_5():
    name = "mcqs_1104_best_5"
    seeds = [0, 1, 2, 3, 4, 5]
    run_nums = [1, 1, 1, 1, 1, 1]
    # seeds = [0,]
    # run_nums = [ 1]
    assert len(run_nums) == len(seeds)
    # seeds = [0, 1,]
    # run_nums = [1, 1,]
    for (seed, run_number) in zip(seeds, run_nums):
        df, _, name = run_experiments.exp_1103_test150_o1mini(seed=seed)
        mcqs, mcqs_question, mcqs_choices, idxs_question, df_results = get_refined_bot_mcqs(
            name, run_number)
        df_results['mcqs'] = mcqs
        df_results_lst.append(df_results)
        assert np.array_equal(df['key_question'].values, idxs_question)

    df_choose_qs, df_results = select_mcqs_by_priority(df_results_lst)
    mcqs = df_choose_qs.values

    return df, df_results, mcqs, name, run_number

def exp_1103_k2_test150_best():
    name = "mcqs_1103_k2_test150_best"
    seeds = [0, 1, 2, 3, 4]
    run_nums = [1, 1, 1, 1, 1]
    seeds = [0,  ]
    run_nums = [1, ]
    assert len(run_nums) == len(seeds)
    # seeds = [0, 1,]
    # run_nums = [1, 1,]
    for (seed, run_number) in zip(seeds, run_nums):
        df, _, name = run_experiments.exp_1103_k2_test150(seed=seed)
        mcqs, mcqs_question, mcqs_choices, idxs_question, df_results = get_refined_bot_mcqs(
            name, run_number)
        df_results['mcqs'] = mcqs
        df_results_lst.append(df_results)
        assert np.array_equal(df['key_question'].values, idxs_question)

    df_choose_qs, df_results = select_mcqs_by_priority(df_results_lst)
    mcqs = df_choose_qs.values

    return df, df_results, mcqs, name, run_number

def exp_1103_test150_multieval_150_best4(multi_eval=3):
    seeds = [0, 1, 2, 3]
    run_nums = [2, 2, 2, 2]
    seeds = [0, ]
    run_nums = [2,]
    for (seed, run_number) in zip(seeds, run_nums):
        df, _, name = run_experiments.exp_1103_test150_multieval_150(seed=seed, multi_eval=multi_eval)
        mcqs, mcqs_question, mcqs_choices, idxs_question, df_results = get_refined_bot_mcqs(
            name, run_number)
        df_results['mcqs'] = mcqs
        df_results_lst.append(df_results)
        assert np.array_equal(df['key_question'].values, idxs_question)

    df_choose_qs, df_results = select_mcqs_by_priority(df_results_lst)
    mcqs = df_choose_qs.values

    return df, df_results, mcqs, name, run_number



def exp_1105_test150_dspy_best():
    seeds = [0, 1, 2, 3,4]
    run_nums = [1,1,1,1]
   
    for (seed, run_number) in zip(seeds, run_nums):
        df, _, name = run_experiments.exp_1105_test150_dspy(seed=seed)
        mcqs, mcqs_question, mcqs_choices, idxs_question, df_results = get_refined_bot_mcqs(
            name, run_number)
        df_results['mcqs'] = mcqs
        df_results_lst.append(df_results)
        assert np.array_equal(df['key_question'].values, idxs_question)

    df_choose_qs, df_results = select_mcqs_by_priority(df_results_lst)
    mcqs = df_choose_qs.values

    name = "exp_1105_test150_dspy_150_best_5"


    return df, df_results, mcqs, name, run_number




if __name__ == "__main__":

    # config of whats in the other script
    df_results_lst = []
    seed_eval = 2

    if 0: 
    # if 1: 
        # df_questions, df_results, mcqs, name, run_number = exp_1103_test150_best_5()
        # df_questions, df_results, mcqs, name, run_number = exp_1103_test150_multieval_150_best4(multi_eval=3)
        # df_questions, df_results, mcqs, name, run_number = exp_1103_test150_o1mini_best_5()
        # df_questions, df_results, mcqs, name, run_number = exp_1103_k2_test150_best()
        df_questions, df_results, mcqs, name, run_number = exp_1105_test150_dspy_best()
        

    else:
        run_number = 1
        seed = 0

        ## a single og experiment
        # df_questions, _, name = run_experiments.exp_1103_test150(seed=seed)
        # df_questions, _, name = run_experiments.exp_1103_test150_o1mini(seed=seed)
        # df_questions, _, name = run_experiments.exp_1103_test150_multieval_150(seed=seed)
        # df_questions, _, name = run_experiments.exp_1105_test150_dspy(seed=seed) # dspy
        # df_questions, _, name = run_experiments.exp_1105_test150_dspy_o1mini(seed=seed) # dspy
        df_questions, _, name = run_experiments.exp_1105_dspy_full(seed=seed) # dspy        
        df_questions, _, name = run_experiments.exp_1105_dspy_full(seed=0) # dspy        

        # get the mcqs
        mcqs, _, _, idxs_question, df_results = get_refined_bot_mcqs(name, run_number)

        # doing the 673 filtering
        # if 1: 
        #     ilocs = [i for (i, idx) in enumerate(df_questions.index) if idx <= 673]
        #     df_questions = df_questions.iloc[ilocs]
        #     df_results = df_results.iloc[ilocs]
        #     mcqs = [mcqs[i] for i in ilocs]
        #     # ipdb.set_trace()
        #     # pass


        df_results['mcqs'] = mcqs
        assert np.array_equal(df_questions['key_question'].values, idxs_question)
        # print("*" * 80)
        # print(f"warning ... idxs_question doesn't quite match {len(df_questions)}")
        # print("*" * 80)


    model = "gpt-4o-2024-08-06"
    do_language_only = True
    # save_cached_images = False
    key_prompt_eval = 0

    df_questions, msgs, preds, gts, cache_images = eval_qa(
        df_questions, mcqs, key_prompt_eval=key_prompt_eval, seed=seed_eval, model=model)
    acc = (gts == preds).sum() / len(gts)
    print(f"Acc VQA {acc:.4f} on {len(gts)} samples")

    df_questions['gt'] = gts
    df_questions['pred'] = preds
    df_questions['pred_correct'] = (preds == gts).astype(int)
    df_questions['pred_cot'] = msgs

    if do_language_only:
        print(f"\n\nRunning language-only:")
        key_prompt_eval = 1
        df_questions, msgs, preds, gts, _ = eval_qa(df_questions,
                                            mcqs,
                                            key_prompt_eval=key_prompt_eval,
                                            seed=seed_eval,
                                            model=model)
        acc = (gts == preds).sum() / len(gts)
        print(f"Acc VQA, no image {acc:.1f}")

        df_questions['pred_no_img'] = preds
        df_questions['pred_no_img_correct'] = (preds == gts).astype(int)
        df_questions['pred_cot_no_img'] = msgs

    # merge the quesitons and results df
    df_results['key_question'] = df_results['key_question'].astype(str)
    df_questions['key_question'] = df_questions['key_question'].astype(str)
    df_eval = df_questions.merge(df_results,
                            on="key_question",
                            suffixes=('_q', '_r'))
    df_eval['mcqs_formatted'] = [json.dumps(m, indent=4) for m in mcqs]

    # save the questions
    df_eval['question_postbot'] = [m['question_stem'] for m in mcqs]
    df_eval['choices_postbot'] = [
        dict(choices=m['choices'], correct_index=m['correct_index'])
        for m in mcqs
    ]

    # save it
    dir_results_parent = Path(f"benchmark/refine_bot/results/eval")
    dir_results_parent.mkdir(exist_ok=True)
    f_save = dir_results_parent / f"{name}_evalseed{seed_eval}.csv"
    print(f"Saving question-level eval to {f_save}")
    df_eval.to_csv(f_save, index=False)
    ipdb.set_trace()
    pass



