"""
python -m ipdb benchmark/final_benchmark/make_final_benchmark.py
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

dir_data = Path(__file__).parent / "data" 
dir_data.mkdir(exist_ok=True)

from benchmark.refine_bot.run_experiments import _download_csv
from benchmark.refine_bot.run_eval import eval_qa

def _combine_data_sheets():
	url_jan26 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5LRmuZWgb20GP6FEGHeB06pnqwV_vuXPK_5CLIl01bYaD8t2SdD5gvIB_dS9SIrmyTAlISEPMkiTZ/pub?gid=2137297679&single=true&output=csv"
	fname = dir_data / 'final_benchmark_jan26.csv'
	if not fname.exists():
		_download_csv(url_jan26, fname)
	df_0 = pd.read_csv(fname)

	
	## get the thing for eval
	df_questions = df_0

	return df_questions


def _lookup_key_person(df_questions, show_name=True):
	
	# recover the `key_person` from earlier data
	df_questions_original = pd.read_csv("benchmark/data/formdata_0/4_questions.csv")
	n_qs = len(df_questions)
	df_questions = pd.merge(df_questions, df_questions_original[['key_question','key_person']])
	assert len(df_questions) == n_qs

	if show_name:
		df_people = pd.read_csv("benchmark/data/formdata_0/2_df_people.csv")
		lookup_person = dict(zip(df_people['key_person'], df_people['Your name']))
		df_questions['person_name'] = [lookup_person[k_p] for k_p in df_questions['key_person']]

	return df_questions

def _generate_mcqs_for_eval(df_questions):
	mcqs = []
	for idx, row in df_questions.iterrows():
		mcq = dict(question_stem=row['question'], correct_index=row['correct_index_2'], choices=ast.literal_eval(row['choices'])['choices'])
		mcqs.append(mcq)

def get_benchmark():
	df_questions = _combine_data_sheets()
	df_questions = _lookup_key_person(df_questions)
	mcqs = _generate_mcqs_for_eval(df_questions)
	df_questions = df_questions.set_index('key_question')
	
	return df_questions, mcqs




if __name__=="__main__":
	df_questions, mcqs = get_benchmark()

	## run eval
	models = ["gpt-4o-2024-08-06", "anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5", "Qwen/Qwen2-VL-72B-Instruct"]
	apis = ["openai", "openrouter", "openrouter", "hyperbolic"]
	models = ['o1-mini-2024-09-12']
	apis = ["openai"]
	models = ["gpt-4o-2024-08-06"]
	apis = ["openai"]
	seed = 10
	key_prompt_eval = 0 

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
		                                            key_prompt_eval=key_prompt_eval,
		                                            model=model,
		                                            api=api,
		                                            num_threads=num_threads,
		                                            seed=seed,
		                                            verbose=False)

		acc = (gts == preds).sum() / len(gts)
		print(f"Acc VQA {acc:.4f} on {len(gts)} samples")
		df_questions['correct'] = (gts == preds).astype(int)

	ipdb.set_trace()
	pass


