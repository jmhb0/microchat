"""
python -m ipdb benchmark/graduate_samples/run_eval.py
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

from benchmark.refine_bot.run_experiments import _download_csv, get_df_from_key
from benchmark.refine_bot.run_eval import eval_qa

dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)


# df_questions = df_stage_1
from benchmark.graduate_samples.combine_dataset import get_full_dataset_before_review, get_naive_choices_data
df_questions, mcqs = get_full_dataset_before_review()
ipdb.set_trace()

seed = 11
seed = 10
key_prompt_eval = 1 # 0 means with images, 1 means without
DO_STAGE1_EVAL = 0
DO_NAIVE_DISTACTOR_GEN = 0
models = ["gpt-4o-2024-08-06", "anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5", "Qwen/Qwen2-VL-72B-Instruct"]
apis = ["openai", "openrouter", "openrouter", "hyperbolic"]
models = ["gpt-4o-2024-08-06", "anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5"]#, "Qwen/Qwen2-VL-72B-Instruct"]
apis = ["openai", "openrouter", "openrouter", "hyperbolic"]
models = ["mistralai/pixtral-large-2411"]
apis = ['openrouter']
models = ['o1-mini-2024-09-12']
apis = ["openai"]
# models = ["Qwen/Qwen2-VL-72B-Instruct",  "google/gemini-pro-1.5"]
# apis = ["hyperbolic",  "openrouter"]
# models = ["Qwen/Qwen2-VL-72B-Instruct"]
# apis = ["hyperbolic"]
# models = ["meta-llama/llama-3.2-90b-vision-instruct", "Qwen/Qwen2-VL-72B-Instruct"]
# apis = [ "openrouter", "hyperbolic"]
# models = ["Qwen/Qwen2-VL-72B-Instruct"]
# apis = ["hyperbolic"]

# models = ["anthropic/claude-3-opus", "gpt-4-turbo-2024-04-09"]
# apis = ["openrouter", "openai"]
# models = ["gpt-4-turbo-2024-04-09"]
# apis = ["openai"]

# models =["Qwen/Qwen2-VL-72B-Instruct"]
# apis = ["hyperbolic"]
# models = ["meta-llama/llama-3.2-11b-vision-instruct"]
# apis = ["openrouter"]
# models =["mistralai/pixtral-12b"]
# apis = ["openrouter"]

# models = ["meta-llama/llama-3.2-90b-vision-instruct"]
# apis = [ "openrouter"]

# models = ["google/gemini-flash-1.5-8b"]
# apis = [ "openrouter"]
# models = ["anthropic/claude-3-haiku", "gpt-4o-mini-2024-07-18"]
# apis = [ "openrouter", "openai"]

# # models = ["gpt-4o-mini-2024-07-18"]
# # apis = [ "openai"]
# models = ["liuhaotian/llava-13b"]
# apis = [ "openrouter"]

# models = ["Qwen/Qwen2-VL-7B-Instruct"]
# apis = ["hyperbolic"]



# # old gpt
# models = ["gpt-4-turbo-2024-04-09"]
# apis = ["openai"]

if not DO_STAGE1_EVAL and not DO_NAIVE_DISTACTOR_GEN:
    log_stage = "stage2"
if DO_STAGE1_EVAL:
    log_stage = "stage1"
    mcqs = [dict(question_stem=q, choices=c, correct_index=i) for (q,c,i) in
        zip(df_questions['question_1'], df_questions['choices_1'], df_questions['correct_index_1'])]
if DO_NAIVE_DISTACTOR_GEN:
    log_stage = "naive"
    df_questions = get_naive_choices_data()
    mcqs = [dict(question_stem=q, choices=c, correct_index=i) for (q,c,i) in
        zip(df_questions['question'], df_questions['choices'], df_questions['correct_index'])]


for model, api in zip(models, apis):
    print(80*"*")
    print(f"Seed {seed}")
    if 'Qwen' in model:
        num_threads = 16
    elif "gpt-4-turbo" in model:
        num_threads = 1
    else:
        num_threads =  32
    df_questions, msgs, preds, gts, _ = eval_qa(df_questions,
                                                mcqs,
                                                key_prompt_eval,
                                                model=model,
                                                api=api,
                                                num_threads=num_threads,
                                                seed=seed,
                                                verbose=False)

    acc = (gts == preds).sum() / len(gts)
    print(f"Acc VQA {acc:.4f} on {len(gts)} samples")
    df_questions['correct'] = (gts == preds).astype(int)
    # accuracy by iteration

    df_question_original = pd.read_csv("benchmark/data/formdata_0/4_questions.csv")
    lookup_use_case = dict(zip(df_question_original['key_question'], df_question_original['use_case']))
    df_questions['use_case'] = [lookup_use_case[k] for k in df_questions['key_question']]
    df_questions['msg'] = msgs
    df_questions['pred'] = preds
    df_questions['gt'] = gts
    print(df_questions.groupby(["use_case"])['correct'].mean())
    
    log_prompt = f"prompt{key_prompt_eval}"
    f_save = dir_results / f"eval_{model.replace('/', '').replace('.','')}_{log_stage}_{log_prompt}.csv"
    print(f"Eval results in {f_save}")
    df_questions.to_csv(f_save)

ipdb.set_trace()
pass
