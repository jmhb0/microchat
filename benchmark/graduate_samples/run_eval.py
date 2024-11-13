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

# stage 1
# df_stage_1 = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

# # yapf: disable
# lookup_dfs = {
#  1 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=0&single=true&output=csv",
#  2 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=1390108749&single=true&output=csv",
#  3 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=1479678577&single=true&output=csv",
#  4 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=426703324&single=true&output=csv",
#  5 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=118252937&single=true&output=csv",
#  6 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=51657708&single=true&output=csv",
#  7 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=1200417443&single=true&output=csv",
#  8 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=1249985140&single=true&output=csv",
#  9 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=1016478403&single=true&output=csv",
# }
# # yapf: enable

# dfs = []

# iter_lst = []
# for iter_ in range(1, 10):
#     url = lookup_dfs[iter_]
#     dir_data = dir_results_parent / "data"
#     dir_data.mkdir(exist_ok=True)
#     output_path = dir_data / f"data_iter{iter_}.csv"
#     _download_csv(url, output_path, overwrite=False)
#     df = pd.read_csv(output_path)
#     dfs.append(df)
#     iter_lst += [iter_]*len(df)

# df = pd.concat(dfs)
# df['iter'] = iter_lst

# # if 0:
# # mcqs = []
# # for i, row in df.iterrows():
# #     choices_postbot = ast.literal_eval(row['choices_postbot'])['choices']
# #     correct_index = ast.literal_eval(
# #         row['choices_postbot'])['correct_index']
# #     mcq = dict(question_stem=row['question_postbot'],
# #                choices=choices_postbot,
# #                correct_index=correct_index)
# #     mcqs.append(mcq)
# # df_questions = df

# df = df.set_index('key_question', drop=False)
# df_stage_1 = df_stage_1.set_index('key_question', drop=False)
# mcqs = []
# df_stage_1['iter'] = 0



# # the full dataset way
# for key_question, row in df_stage_1.iterrows():
#     if key_question in df.index:
#         row_postpot = df.loc[key_question]
#         df_stage_1.loc[key_question, 'iter'] = int(row_postpot['iter'])
#         row['iter'] = int(row_postpot['iter'])
#         choices_postbot = ast.literal_eval(
#             row_postpot['choices_postbot'])['choices']
#         correct_index = ast.literal_eval(
#             row_postpot['choices_postbot'])['correct_index']
#         mcq = dict(question_stem=row_postpot['question_postbot'],
#                    choices=choices_postbot,
#                    correct_index=correct_index)
#     else:
#         row_s1 = df_stage_1.loc[key_question]
#         mcq = dict(question_stem=row_s1['question'],
#                    choices=row_s1['options'],
#                    correct_index=row_s1['correct_index'])

#     mcqs.append(mcq)

# df_questions = df_stage_1
from benchmark.graduate_samples.combine_dataset import get_full_dataset_before_review, get_naive_choices_data
df_questions, mcqs = get_full_dataset_before_review()

seed = 11
seed = 10
models = ["gpt-4o-2024-08-06", "anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5", "Qwen/Qwen2-VL-72B-Instruct"]
apis = ["openai", "openrouter", "openrouter", "hyperbolic"]
# models = ["Qwen/Qwen2-VL-72B-Instruct",  "google/gemini-pro-1.5"]
# apis = ["hyperbolic",  "openrouter"]
# models = ["Qwen/Qwen2-VL-72B-Instruct"]
# apis = ["hyperbolic"]
# models = ["meta-llama/llama-3.2-90b-vision-instruct"]
# apis = [ "openrouter"]

# ### experiment on the mcqs stage 1
# models = ["anthropic/claude-3.5-sonnet"]
# apis = ["openrouter"]

# models = [ "google/gemini-pro-1.5"]
# apis = [ "openrouter"]

# models = ["gpt-4o-2024-08-06"]
# apis = ["openai"]

# # old gpt
models = ["gpt-4-turbo-2024-04-09"]
apis = ["openai"]

log_stage = "stage2"
DO_STAGE1_EVAL = 0
if DO_STAGE1_EVAL:
    log_stage = "stage1"
    mcqs = [dict(question_stem=q, choices=c, correct_index=i) for (q,c,i) in
        zip(df_questions['question_1'], df_questions['choices_1'], df_questions['correct_index_1'])]
DO_NAIVE_DISTACTOR_GEN = 0
if DO_NAIVE_DISTACTOR_GEN:
    log_stage = "naive"
    df_questions = get_naive_choices_data()
    mcqs = [dict(question_stem=q, choices=c, correct_index=i) for (q,c,i) in
        zip(df_questions['question'], df_questions['choices'], df_questions['correct_index'])]


for model, api in zip(models, apis):
    # model = "gpt-4o-2024-08-06"; api = "openai"
    # model="anthropic/claude-3.5-sonnet"; api="openrouter"
    # model="google/gemini-pro-1.5"; api="openrouter"
    # model = "Qwen/Qwen2-VL-72B-Instruct"; api = "hyperbolic"
    print(80*"*")
    print(f"Seed {seed}")
    key_prompt_eval = 0
    num_threads =  32 if 'Qwen' not in 'model' else 8 # 'hyperbolic' api has rate limiting
    num_threads = 1
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
    
    f_save = dir_results / f"eval_{model.replace('/', '').replace('.','')}_{log_stage}.csv"
    print(f"Eval results in {f_save}")
    df_questions.to_csv(f_save)

