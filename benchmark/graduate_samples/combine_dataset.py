"""
python -m ipdb benchmark/graduate_samples/combine_dataset.py

For getting the 
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

dir_results_parent = Path(__file__).parent / "results" / Path(__file__).stem
dir_results_parent.mkdir(exist_ok=True, parents=True)

def get_full_dataset_before_review():
	# stage 1
	df_stage_1 = get_df_from_key("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)
	df_stage_1 = df_stage_1.set_index("key_question", drop=False)

	# stage 2 round 1
	# yapf: disable
	lookup_dfs_rnd1 = {
	 # 1 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=0&single=true&output=csv",
	 2 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=1390108749&single=true&output=csv",
	 3 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=1479678577&single=true&output=csv",
	 4 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=426703324&single=true&output=csv",
	 5 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=118252937&single=true&output=csv",
	 6 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=51657708&single=true&output=csv",
	 7 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=1200417443&single=true&output=csv",
	 8 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=1249985140&single=true&output=csv",
	 9 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vS2hKctARf4rPu3JqRLDaJB7kw5oM9_1EKRIxs3Y8NcP4XydozenDxNME51Gqxv-N50xDt_xSvF4o68/pub?gid=1016478403&single=true&output=csv",
	}
	# yapf: enable
	iter_lst = []
	dfs = []
	for iter_ in range(2, 10):
	    url = lookup_dfs_rnd1[iter_]
	    dir_data = dir_results_parent / "data"
	    dir_data.mkdir(exist_ok=True)
	    output_path = dir_data / f"data_iter{iter_}.csv"
	    _download_csv(url, output_path, overwrite=False)
	    df = pd.read_csv(output_path)
	    dfs.append(df)
	    iter_lst += [f"1.{iter_}"]*len(df)
	df_rnd1 = pd.concat(dfs)
	df_rnd1['iter'] = iter_lst
	# yapf: disable

	# stage 2 round 2
	lookup_dfs_rnd2 = {
	 1 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTjBEY-n9KddEkKXQJZQfh-DfGG-5EWEQjJs-as8AzAWqPonsvcWCtqllfE_qrjjHSwMWQe95yVGMsu/pub?gid=771370350&single=true&output=csv",
	 2 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTjBEY-n9KddEkKXQJZQfh-DfGG-5EWEQjJs-as8AzAWqPonsvcWCtqllfE_qrjjHSwMWQe95yVGMsu/pub?gid=1754405408&single=true&output=csv",
	 3 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTjBEY-n9KddEkKXQJZQfh-DfGG-5EWEQjJs-as8AzAWqPonsvcWCtqllfE_qrjjHSwMWQe95yVGMsu/pub?gid=617768050&single=true&output=csv",
	 4 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTjBEY-n9KddEkKXQJZQfh-DfGG-5EWEQjJs-as8AzAWqPonsvcWCtqllfE_qrjjHSwMWQe95yVGMsu/pub?gid=2017256061&single=true&output=csv",
	 5 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTjBEY-n9KddEkKXQJZQfh-DfGG-5EWEQjJs-as8AzAWqPonsvcWCtqllfE_qrjjHSwMWQe95yVGMsu/pub?gid=729997762&single=true&output=csv",
	 6 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTjBEY-n9KddEkKXQJZQfh-DfGG-5EWEQjJs-as8AzAWqPonsvcWCtqllfE_qrjjHSwMWQe95yVGMsu/pub?gid=170539827&single=true&output=csv",
	}
	# yapf: enable
	iter_lst = []
	dfs = []
	for iter_ in range(1, 7):
	    url = lookup_dfs_rnd2[iter_]
	    dir_data = dir_results_parent / "data"
	    dir_data.mkdir(exist_ok=True)
	    output_path = dir_data / f"data_rnd2_iter_{iter_}.csv"
	    _download_csv(url, output_path, overwrite=False)
	    df = pd.read_csv(output_path)
	    dfs.append(df)
	    iter_lst += [f"2.{iter_}"]*len(df)
	df_rnd2 = pd.concat(dfs)
	df_rnd2['iter'] = iter_lst
	df_rnd2 = df_rnd2.groupby('key_question', as_index=False).first()

	# combine
	df_stage_2 = pd.concat([df_rnd1, df_rnd2])
	df_stage_2 = df_stage_2.set_index('key_question', drop=False)

	# construct the full dataset
	mcqs = []
	for key_question, row in df_stage_1.iterrows():

		# case 1: stage 2 did work, then use it
	    if key_question in df_stage_2.index:
	        row_postpot = df_stage_2.loc[key_question]
	        df_stage_1.loc[key_question, 'iter'] = str(row_postpot['iter'])
	        row['iter'] = str(row_postpot['iter'])
	        choices_postbot = ast.literal_eval(
	            row_postpot['choices_postbot'])['choices']
	        correct_index = ast.literal_eval(
	            row_postpot['choices_postbot'])['correct_index']
	        mcq = dict(question_stem=row_postpot['question_postbot'],
	                   choices=choices_postbot,
	                   correct_index=correct_index)
	    # otherwise fallback
	    else:
	        row_s1 = df_stage_1.loc[key_question]
	        mcq = dict(question_stem=row_s1['question'],
	                   choices=row_s1['options'],
	                   correct_index=row_s1['correct_index'])

	    mcqs.append(mcq)

	df_questions = df_stage_1

	return df_questions, mcqs


if __name__=="__main__":
	df_questions, mcqs = get_full_dataset_before_review()




