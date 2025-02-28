"""
python -m ipdb benchmark/refine_bot/run_experiments.py
"""
import ipdb
import time
from pathlib import Path
import pandas as pd
import requests
import json
import ast
import re

from benchmark.refine_bot import bot

### first define some common configs
n_choices_target = 5

model_o1 = "o1-preview-2024-09-12"
model_o1mini = "o1-mini-2024-09-12"
model_gpt4o = "gpt-4o-2024-08-06"
model_gpt4omini = "gpt-4o-mini-2024-07-18"

# yapf: disable
lookup_dfs = {
 "1103_naive_kq3_kc9": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTf4Xzjcosbjdt12M_AyGLP4UimHXZ6uEGK7WDdkAg97ErKuBswkXkmr55CEhMWl3R8FUlEap0AS-1P/pub?gid=1999709323&single=true&output=csv",
 "dspy_o1-mini_CoTRAG" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTf4Xzjcosbjdt12M_AyGLP4UimHXZ6uEGK7WDdkAg97ErKuBswkXkmr55CEhMWl3R8FUlEap0AS-1P/pub?gid=2096678925&single=true&output=csv",
 "dspy_o1-mini_CoTRAG_FULL_nov5" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTf4Xzjcosbjdt12M_AyGLP4UimHXZ6uEGK7WDdkAg97ErKuBswkXkmr55CEhMWl3R8FUlEap0AS-1P/pub?gid=1953831301&single=true&output=csv",
 # this is the last 50 or so quesitons
 "feb5" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5VP4RdJqTkleK-AGVUTPMAzAeM64IsMawG_w5ghSnvqQ_2hfEjPBZcUIKplzS7qeJKEVUVxSiazdh/pub?gid=1858657171&single=true&output=csv",
}

max_iters = 3
multi_eval = 1
cfg_4o_k0 = dict(
 name="cfg_4o_k0",
 seed=0,
 max_iters=max_iters,
 eval=dict(model=model_gpt4o, key=0, multi_eval=multi_eval),
 reflect=dict(model=model_gpt4o, key=0),
 rewrite=dict(model=model_gpt4o, key=0, strucured_output_key=1, n_choices_target=n_choices_target),
 check_rewrite=dict(model=model_gpt4o, key=0, strucured_output_key=1),
)
# key 1 gpt 4o
cfg_4o_k1 = dict(
 name="cfg_4o_k1",
 seed=0,
 max_iters=max_iters,
 eval=dict(model=model_gpt4o, key=1, multi_eval=multi_eval),
 reflect=dict(model=model_gpt4o, key=1),
 rewrite=dict(model=model_gpt4o, key=1, strucured_output_key=1, n_choices_target=n_choices_target),
 check_rewrite=dict(model=model_gpt4o, key=1, strucured_output_key=1),
)
# key 2 gpt 4o. the rewrite key is still
cfg_4o_k2 = dict(
 name="cfg_4o_k2",
 seed=0,
 max_iters=max_iters,
 eval=dict(model=model_gpt4o, key=2, multi_eval=multi_eval),
 reflect=dict(model=model_gpt4o, key=2),
 rewrite=dict(model=model_gpt4o, key=2, strucured_output_key=1, n_choices_target=n_choices_target),
 check_rewrite=dict(model=model_gpt4o, key=1, strucured_output_key=1),
)
cfg_o1_k1 = dict(
 name="key1-modelo1",
 seed=1,
 max_iters=max_iters,
 eval=dict(model=model_o1, key=1, multi_eval=multi_eval),
 reflect=dict(model=model_o1, key=1),
 rewrite=dict(model=model_o1, key=1, strucured_output_key=0, n_choices_target=5),
 check_rewrite=dict(model=model_o1, key=1, strucured_output_key=0),
)
cfg_o1_k0 = dict(
 name="cfg_o1_k0",
 seed=0,
 max_iters=max_iters,
 eval=dict(model=model_o1, key=0, multi_eval=multi_eval),
 reflect=dict(model=model_o1, key=0),
 rewrite=dict(model=model_o1, key=0, strucured_output_key=0, n_choices_target=n_choices_target),
 check_rewrite=dict(model=model_gpt4o, key=0, strucured_output_key=1),
)
cfg_o1mini_k1 = dict(
 name="key1-modelo1mini",
 seed=0,
 max_iters=max_iters,
 eval=dict(model=model_o1mini, key=1, multi_eval=multi_eval),
 reflect=dict(model=model_o1mini, key=1),
 rewrite=dict(model=model_o1mini, key=1, strucured_output_key=0, n_choices_target=5),
 check_rewrite=dict(model=model_o1mini, key=1, strucured_output_key=0),
)
# yapf: enable


def get_df_from_key(key, overwrite=False):
	url = lookup_dfs[key]

	dir_csvs = Path(__file__).parent / "results/dfs"
	dir_csvs.mkdir(exist_ok=True)
	f_csv = dir_csvs / f"{key}.csv"
	_download_csv(url, f_csv, overwrite=overwrite)
	df = pd.read_csv(f_csv)

	# align the dspy name alignment with the new one - these were from stage 1 
	if 'dspy' in key:
		assert 'question' not in df.columns
		assert 'choices' not in df.columns
		# df['question_original'] = df['question'].copy()

		df['question'] = df['revised_question']
		assert [s[:10] == "Question\n" for s in df['revised_question']]
		df['question'] = [s[10:] for s in df['revised_question']]
		df['revised_question'].iloc[0][:10]

		df['options'] = [ast.literal_eval(op) for op in df['options']]
		df['choices'] = [{
			'choices': ops,
			'correct_index': c
		} for (ops, c) in zip(df['options'], df['correct_index'])]
		df['choices'] = [json.dumps(c) for c in df['choices']]
		df['use_case'] = -1

	df['key_question'] = df['key_question'].astype(int)
	df['key_image'] = df['key_image'].astype(int)

	# these are stage 1 format 
	if key == "feb5":
		# Split the revised_question_answer into components
		splits = [s.split("\n\n") for s in df['revised_question_answer']]
		
		# Extract question, options, and answer
		questions = ["\n\n".join(parts[:-2]) for parts in splits]
		# Remove the "Question:\n```" prefix
		df['question'] = [q.replace("Question:\n```", "").strip() for q in questions]
		
		# Parse options into list format
		options = [parts[-2].split("\n") for parts in splits]
		df['options'] = [[opt[3:].strip() for opt in option_group] for option_group in options] # remove the leading A. or B. or whatver

		# Get correct answer and convert to index (A=0, B=1, etc)
		answer_letters = [parts[-1][16] for parts in splits]
		# Create explicit mapping from letters to indices
		letter_to_index = {letter: idx for idx, letter in enumerate(list('ABCDE'))}
		
		# Map answer letters to indices with explicit error if letter not found
		df['correct_index'] = [letter_to_index[ans] for ans in answer_letters]
		
		# Create choices column in the same format as dspy
		df['choices'] = [{
			'choices': ops,
			'correct_index': c
		} for (ops, c) in zip(df['options'], df['correct_index'])]
		df['choices'] = [json.dumps(c) for c in df['choices']]
		
		df['use_case'] = -1

	return df


def _download_csv(url, output_path, overwrite=False):
	if overwrite or not Path(output_path).exists():
		response = requests.get(url)
		response.raise_for_status()
		with open(output_path, 'wb') as f:
			f.write(response.content)
		print("Downloading CSV from:", output_path)
	else:
		print("Retrieved cached CSV from:", output_path)


def _get_key_questions_sample():
	f_idxs_sample = "benchmark/refine_bot/idxs_sample.json"

	with open(f_idxs_sample, 'r') as fp:
		idxs = json.load(fp)

	return idxs


def exp_1102_first150():
	"""
	The first 150 as a list 
	"""
	name = "exp_1102_first150"
	df, do_shuffle = _get_data_october('qkey3_ckey9')
	cfg = cfg_4o_k1
	df = df.iloc[:150]

	return df, cfg, name


def exp_1103_test150(seed):
	""" The random sample 150 """
	# configs
	name = f"exp_1103_test150_seed_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed

	# get dataset
	df = get_df_from_key("1103_naive_kq3_kc9", overwrite=False)
	keys_question = _get_key_questions_sample()
	df = df[df['key_question'].isin(keys_question)]

	return df, cfg, name


def exp_1103_k2_test150(seed):
	""" The random sample 150 """
	# configs
	name = f"exp_1103_k2_test150_seed_{seed}"
	cfg = cfg_4o_k2
	cfg['seed'] = seed

	# get dataset
	df = get_df_from_key("1103_naive_kq3_kc9", overwrite=False)
	keys_question = _get_key_questions_sample()
	df = df[df['key_question'].isin(keys_question)]
	return df, cfg, name


def exp_1103_test150_o1mini(seed):
	df, cfg, name = exp_1103_test150(seed)
	name = "cfg_4o_k1_"

	for k in ('eval', 'reflect', 'rewrite', 'check_rewrite'):
		cfg[k]['model'] = model_gpt4omini
		raise "wrong model specified, look at exp_1105_test150_dspy_o1mini"
	name += f"_o1mini_{seed}"

	return df, cfg, name


def exp_1103_test150_multieval_150(seed, multi_eval=3):
	""" 
	The random sample 150. 
	"""
	# configs
	name = f"exp_1103_test150multieval{multi_eval}_seed_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	cfg['eval']['multi_eval'] = multi_eval

	# get dataset
	df = get_df_from_key("1103_naive_kq3_kc9", overwrite=False)
	keys_question = _get_key_questions_sample()
	df = df[df['key_question'].isin(keys_question)]

	return df, cfg, name


def exp_1105_test150_dspy(seed):
	""" The random sample 150 """
	# configs
	name = f"exp_1105_test150_dspy_seed_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG", overwrite=False)
	keys_question = _get_key_questions_sample()
	df = df[df['key_question'].isin(keys_question)]

	return df, cfg, name


def exp_1105_dspy_full(seed):
	""" The full dataset coming out of stage 1 dspy optimiztion process. """
	# configs
	name = f"dspy_full_nov5_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	return df, cfg, name


def exp_1105_test150_dspy_o1mini(seed):
	""" The random sample 150 """
	# configs
	name = f"exp_1105_test150_dspy_o1mini_seed_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	for k in ('eval', 'reflect', 'rewrite', 'check_rewrite'):
		cfg[k]['model'] = model_o1mini
	for k in ('rewrite', 'check_rewrite'):
		cfg[k]['strucured_output_key'] = 0

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG", overwrite=False)
	keys_question = _get_key_questions_sample()
	df = df[df['key_question'].isin(keys_question)]

	return df, cfg, name


def exp_1109_dspy_full_after_round1(seed):
	""" The full dataset coming out of stage 1 dspy optimiztion process. """
	# configs
	name = f"exp_exp_1109_dspy_full_iter2_{seed}"
	# cfg = cfg_4o_k1
	cfg['seed'] = seed

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	# eliminate some keys
	with open("benchmark/refine_bot/keys_round1.json", "r") as fp:
		keys_round1 = json.load(fp)

	n_before = len(df)
	n_after = n_before - len(keys_round1)
	df = df[~df['key_question'].isin(keys_round1)].copy()
	assert len(df) == n_after

	return df, cfg, name


def exp_1109_dspy_full_after_round2_o1mini(seed):
	"""
	This one does o1mini 
	"""
	# configs
	name = f"exp_1109_dspy_full_after_round2_o1mini_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	# for k in ('eval', 'reflect', 'rewrite', 'check_rewrite'):
	cfg['rewrite']['model'] = model_o1mini
	cfg['rewrite']['strucured_output_key'] = 0

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	# eliminate some keys
	with open("benchmark/refine_bot/keys_round1.json", "r") as fp:
		keys_round1 = json.load(fp)
	with open("benchmark/refine_bot/keys_round2.json", "r") as fp:
		keys_round2 = json.load(fp)

	keys_skip = keys_round1 + keys_round2

	n_before = len(df)
	n_after = n_before - len(keys_skip)
	df = df[~df['key_question'].isin(keys_skip)].copy()
	assert len(df) == n_after

	return df, cfg, name


def exp_1109_dspy_full_after_round3_o1mini(seed):
	"""
	This one does o1mini again 
	"""
	# configs
	name = f"exp_1109_dspy_full_after_round3_o1mini_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	# for k in ('eval', 'reflect', 'rewrite', 'check_rewrite'):
	cfg['rewrite']['model'] = model_o1mini
	cfg['rewrite']['strucured_output_key'] = 0

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	# eliminate some keys
	with open("benchmark/refine_bot/keys_round1.json", "r") as fp:
		keys_round1 = json.load(fp)
	with open("benchmark/refine_bot/keys_round2.json", "r") as fp:
		keys_round2 = json.load(fp)
	with open("benchmark/refine_bot/keys_round3.json", "r") as fp:
		keys_round3 = json.load(fp)

	keys_skip = keys_round1 + keys_round2 + keys_round3

	n_before = len(df)
	n_after = n_before - len(keys_skip)
	df = df[~df['key_question'].isin(keys_skip)].copy()
	assert len(df) == n_after

	return df, cfg, name


def exp_1109_dspy_full_after_round4_o1(seed):
	"""
	This one does o1mini again 
	"""
	# configs
	name = f"exp_1109_dspy_full_after_round4_o1_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	# for k in ('eval', 'reflect', 'rewrite', 'check_rewrite'):
	cfg['rewrite']['model'] = model_o1
	cfg['rewrite']['strucured_output_key'] = 0
	cfg['max_iters'] = 2  # because it's expensive

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	# eliminate some keys
	with open("benchmark/refine_bot/keys_round1.json", "r") as fp:
		keys_round1 = json.load(fp)
	with open("benchmark/refine_bot/keys_round2.json", "r") as fp:
		keys_round2 = json.load(fp)
	with open("benchmark/refine_bot/keys_round3.json", "r") as fp:
		keys_round3 = json.load(fp)
	with open("benchmark/refine_bot/keys_round4.json", "r") as fp:
		keys_round4 = json.load(fp)

	keys_skip = keys_round1 + keys_round2 + keys_round3 + keys_round4

	n_before = len(df)
	n_after = n_before - len(keys_skip)
	df = df[~df['key_question'].isin(keys_skip)].copy()
	assert len(df) == n_after

	return df, cfg, name


def exp_1109_dspy_full_after_round5_4oagain(seed):
	"""
	Here we call 4o again, just on the filtered versions - this should just have
	cache hits
	"""
	# configs
	name = f"exp_1109_dspy_full_after_round5_4oagain_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	# for k in ('eval', 'reflect', 'rewrite', 'check_rewrite'):
	# cfg['rewrite']['model'] = model_o1
	# cfg['rewrite']['strucured_output_key'] = 0
	cfg['max_iters'] = 3  # because it's expensive

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	# eliminate some keys
	with open("benchmark/refine_bot/keys_round1.json", "r") as fp:
		keys_round1 = json.load(fp)
	with open("benchmark/refine_bot/keys_round2.json", "r") as fp:
		keys_round2 = json.load(fp)
	with open("benchmark/refine_bot/keys_round3.json", "r") as fp:
		keys_round3 = json.load(fp)
	with open("benchmark/refine_bot/keys_round4.json", "r") as fp:
		keys_round4 = json.load(fp)
	with open("benchmark/refine_bot/keys_round5.json", "r") as fp:
		keys_round5 = json.load(fp)

	keys_skip = keys_round1 + keys_round2 + keys_round3 + keys_round4 + keys_round5

	n_before = len(df)
	n_after = n_before - len(keys_skip)
	df = df[~df['key_question'].isin(keys_skip)].copy()
	assert len(df) == n_after

	return df, cfg, name


def exp_1110_redo_4o_fromiter1_iter1(seed):
	"""
	Here we call 4o again, just on the filtered versions - this should just have
	cache hits
	"""
	# configs
	name = f"exp_1110_redo_4o_fromiter1_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	cfg['max_iters'] = 3  # because it's expensive

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	# eliminate some keys
	with open("benchmark/refine_bot/keys_round1.json", "r") as fp:
		keys_round1 = json.load(fp)

	df = df[df['key_question'].isin(keys_round1)].copy()
	assert len(df) == len(keys_round1)

	return df, cfg, name

def exp_1110_redo_4o_fromiter1_iter2(seed):
	"""
	Here we call 4o again, just on the filtered versions - this should just have
	cache hits
	"""
	# configs
	name = f"exp_1110_redo_4o_fromiter2_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	cfg['max_iters'] = 3  # because it's expensive

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	# eliminate some keys
	with open("benchmark/refine_bot/keys_round1.json", "r") as fp:
		keys_round1 = json.load(fp)

	df = df[df['key_question'].isin(keys_round1)].copy()
	assert len(df) == len(keys_round1)

	# now filter the keys we already got
	with open("benchmark/refine_bot/keys_redo_round1.json", "r") as fp:
		keys_redo_round1 = json.load(fp)

	keys_skip = keys_redo_round1

	n_before = len(df)
	n_after = n_before - len(keys_skip)
	df = df[~df['key_question'].isin(keys_skip)].copy()
	assert len(df) == n_after

	return df, cfg, name


def exp_1110_redo_4o_fromiter1_iter3(seed):
	"""
	Here we call 4o again, just on the filtered versions - this should just have
	cache hits
	"""
	# configs
	name = f"exp_1110_redo_4o_fromiter3_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	cfg['max_iters'] = 3  # because it's expensive
	cfg['rewrite']['model'] = model_o1mini
	cfg['rewrite']['strucured_output_key'] = 0

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	# eliminate some keys
	with open("benchmark/refine_bot/keys_round1.json", "r") as fp:
		keys_round1 = json.load(fp)

	df = df[df['key_question'].isin(keys_round1)].copy()
	assert len(df) == len(keys_round1)

	# now filter the keys we already got
	with open("benchmark/refine_bot/keys_redo_round1.json", "r") as fp:
		keys_redo_round1 = json.load(fp)
	with open("benchmark/refine_bot/keys_redo_round2.json", "r") as fp:
		keys_redo_round2 = json.load(fp)

	keys_skip = keys_redo_round1 + keys_redo_round2

	n_before = len(df)
	n_after = n_before - len(keys_skip)
	df = df[~df['key_question'].isin(keys_skip)].copy()
	assert len(df) == n_after

	return df, cfg, name

def exp_1110_redo_4o_fromiter1_iter4(seed):
	"""
	Here we call 4o again, just on the filtered versions - this should just have
	cache hits
	"""
	# configs
	name = f"exp_1110_redo_4o_fromiter4_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	cfg['max_iters'] = 3  # because it's expensive
	# cfg['rewrite']['model'] = model_o1mini
	# cfg['rewrite']['strucured_output_key'] = 0

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	# eliminate some keys
	with open("benchmark/refine_bot/keys_round1.json", "r") as fp:
		keys_round1 = json.load(fp)

	df = df[df['key_question'].isin(keys_round1)].copy()
	assert len(df) == len(keys_round1)

	# now filter the keys we already got
	with open("benchmark/refine_bot/keys_redo_round1.json", "r") as fp:
		keys_redo_round1 = json.load(fp)
	with open("benchmark/refine_bot/keys_redo_round2.json", "r") as fp:
		keys_redo_round2 = json.load(fp)
	with open("benchmark/refine_bot/keys_redo_round3.json", "r") as fp:
		keys_redo_round3 = json.load(fp)

	keys_skip = keys_redo_round1 + keys_redo_round2 + keys_redo_round3 

	n_before = len(df)
	n_after = n_before - len(keys_skip)
	df = df[~df['key_question'].isin(keys_skip)].copy()
	assert len(df) == n_after

	return df, cfg, name

def exp_1110_redo_4o_fromiter1_iter4_b(seed):
	"""
	Here we call 4o again, just on the filtered versions - this should just have
	cache hits
	"""
	# configs
	name = f"exp_1110_redo_4o_fromiter1_iter4_b_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	cfg['max_iters'] = 3  # because it's expensive
	cfg['rewrite']['model'] = model_o1mini
	cfg['rewrite']['strucured_output_key'] = 0

	# get dataset
	df = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)

	# eliminate some keys
	with open("benchmark/refine_bot/keys_round1.json", "r") as fp:
		keys_round1 = json.load(fp)

	df = df[df['key_question'].isin(keys_round1)].copy()
	assert len(df) == len(keys_round1)

	# now filter the keys we already got
	with open("benchmark/refine_bot/keys_redo_round1.json", "r") as fp:
		keys_redo_round1 = json.load(fp)
	with open("benchmark/refine_bot/keys_redo_round2.json", "r") as fp:
		keys_redo_round2 = json.load(fp)
	with open("benchmark/refine_bot/keys_redo_round3.json", "r") as fp:
		keys_redo_round3 = json.load(fp)

	keys_skip = keys_redo_round1 + keys_redo_round2 + keys_redo_round3 

	n_before = len(df)
	n_after = n_before - len(keys_skip)
	df = df[~df['key_question'].isin(keys_skip)].copy()
	assert len(df) == n_after

	return df, cfg, name


def _exp_0207_round1(seed):
	"""
	"""
	# configs
	name = f"_exp_0207_test_{seed}"
	cfg = cfg_4o_k1
	cfg['seed'] = seed
	cfg['max_iters'] = 3  # because it's expensive
	cfg['rewrite']['model'] = model_gpt4o
	cfg['rewrite']['strucured_output_key'] = 1

	# get dataset
	df = get_df_from_key("feb5", overwrite=True)

	return df, cfg, name


import tiktoken
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a string using tiktoken.
    
    Args:
        text (str): The input text to count tokens for
        model (str): The model to use for tokenization (default: "gpt-3.5-turbo")
        
    Returns:
        int: Number of tokens in the text
    """
    text = re.sub(r'Answer.*$', 'Answer', text)

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    return len(tokens)



def _get_data_october(questions_source):
	""" 
	These were the data sources from earlier experiments
	"""
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


if __name__ == "__main__":
	dir_results_parent = Path(__file__).parent / "results" / Path(
		__file__).stem
	dir_results_parent.mkdir(exist_ok=True, parents=True)
	# do_multiprocessing = False
	do_multiprocessing = True

	## run this experiment
	# df, cfg, name = exp_1103_test150(seed=0)

	# df, cfg, name = exp_1103_test150(seed=1)
	# df, cfg, name = exp_1103_test150(seed=2)
	# df, cfg, name = exp_1103_test150(seed=3)
	# df, cfg, name = exp_1103_test150(seed=4)
	# df, cfg, name = exp_1103_test150_o1mini(seed=0)

	# if 1:
	# for seed in [0]:
	# for seed in [0]:
	for seed in [0, 1, 2, 3, 4, 5, 6]:
		# df, cfg, name = exp_1105_test150_dspy_o1mini(seed=seed)

		# df, cfg, name = exp_1103_test150_multieval_150(seed=seed, multi_eval=3)
		# df, cfg, name = exp_1103_test150_o1mini(seed=seed)
		# df, cfg, name = exp_1103_k2_test150(seed=seed)
		# df, cfg, name = exp_1105_test150_dspy(seed=seed)
		# df, cfg, name = exp_1105_dspy_full(seed=seed)

		# df, cfg, name = exp_1109_dspy_full_after_round1(seed=seed)
		# df, cfg, name = exp_1109_dspy_full_after_round1(seed=seed)
		# df, cfg, name = exp_1109_dspy_full_after_round4_o1(seed=seed)
		# df, cfg, name = exp_1109_dspy_full_after_round5_4oagain(seed=seed)
		# df, cfg, name = exp_1110_redo_4o_fromiter1_iter1(seed=seed)
		# df, cfg, name = exp_1110_redo_4o_fromiter1_iter3(seed=seed)
		# df, cfg, name = exp_1110_redo_4o_fromiter1_iter4(seed=seed)
		# df, cfg, name = exp_1110_redo_4o_fromiter1_iter4_b(seed=seed)
		# ipdb.set_trace()
		# pass

		df, cfg, name = _exp_0207_round1(seed=seed)
		dir_results = dir_results_parent / f"{name}"
		dir_results.mkdir(exist_ok=True, parents=True)
		# df = df.iloc[:25]

		bot.run(df,
				cfg,
				dir_results,
				do_multiprocessing=do_multiprocessing,
				do_shuffle=False)
	# ipdb.set_trace()
	# pass
