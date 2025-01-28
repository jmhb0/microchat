"""
python -m ipdb analysis_scripts/20250126_prepare_human_eval_forms.py

"""

import ipdb
import sys
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from benchmark.final_benchmark.make_final_benchmark import get_benchmark

dir_results_parent = Path(__file__).parent / "results" / Path(__file__).stem
dir_results_parent.mkdir(exist_ok=True, parents=True)

df_questions, mcqs = get_benchmark()

def assign_questions():
	# Conner, Jesus, Ridhi, Disha, Will, Alex, Jan
	n_samples = 100
	random_state=0
	keys_taken = []

	# ipdb> df_questions['person_name'].unique()
	# array(['Sarina Hasan', 'Chad Liu', 'Jan Niklas Hansen', 'Zachary Coman',
	#        'Disha Bhowmik ', 'Will Leineweber', 'Ridhi Yarlagadda',
	#        'Malvika Nair', 'Alexandra Johannesson', 'Jesus G. Galaz-Montoya',
	#        'Alexandra', 'Connor Zuraski', 'Zach Coman'], dtype=object)

	# connor
	keys_this_person = df_questions[df_questions['person_name']=='Connor Zuraski'].index.values
	df_connor_= df_questions[~df_questions.index.isin(keys_taken+list(keys_this_person))]
	df_connor = df_connor_.sample(n=n_samples, random_state=random_state)
	keys_taken += list(df_connor.index)

	# jesus
	keys_this_person = df_questions[df_questions['person_name']=='Jesus G. Galaz-Montoya'].index.values
	df_jesus_ = df_questions[~df_questions.index.isin(keys_taken+list(keys_this_person))]
	df_jesus = df_jesus_.sample(n=n_samples, random_state=random_state)
	keys_taken += list(df_jesus.index)

	# ridhi
	keys_this_person = df_questions[df_questions['person_name']=='Ridhi Yarlagadda'].index.values
	df_ridhi_ = df_questions[~df_questions.index.isin(keys_taken+list(keys_this_person))]
	df_ridhi = df_ridhi_.sample(n=n_samples, random_state=random_state)
	keys_taken += list(df_ridhi.index)

	# disha
	keys_this_person = df_questions[df_questions['person_name']=='Disha Bhowmik'].index.values
	df_disha_ = df_questions[~df_questions.index.isin(keys_taken+list(keys_this_person))]
	df_disha = df_disha_.sample(n=n_samples, random_state=random_state)
	keys_taken += list(df_disha.index)

	# will
	keys_this_person = df_questions[df_questions['person_name']=='Will Leineweber'].index.values
	df_will_ = df_questions[~df_questions.index.isin(keys_taken+list(keys_this_person))]
	df_will = df_will_.sample(n=n_samples, random_state=random_state)
	keys_taken += list(df_will.index)

	# alex - we are already at 500, so she just gets whatever is left
	keys_this_person = df_questions[df_questions['person_name']=='Alexandra Johannesson'].index.values
	df_alex_ = df_questions[~df_questions.index.isin(keys_taken+list(keys_this_person))]
	df_alex = df_alex_
	# df_alex = df_alex_.sample(n=n_samples, random_state=random_state)
	keys_taken += list(df_alex.index)


	dfs = [df_connor, df_jesus, df_ridhi, df_disha, df_will, df_alex]
	dfs = [df.sort_index() for df in dfs]
	names = ['connor', 'jesus', 'ridhi', 'disha', 'will', 'alex']

	return dfs, names

df_all, names = assign_questions()

letters=list('abcdefg')
for (name, df) in zip(names, df_all):
	mcqs = []
	for i, row in df.iterrows():
		mcq = row['question'] + "\n\n"
		choices = ast.literal_eval(row['choices'])['choices']
		for letter, choice in zip(letters, choices):
			mcq += f"({letter}) {choice}\n"
		mcqs.append(mcq)

	df['mcq_str'] = mcqs
	dir_results_person = dir_results_parent / name
	dir_results_person.mkdir(exist_ok=True)
	df.to_csv(dir_results_person / "data.csv")

	key_imgs = df['key_image'].values
	dir_img_grids = "analysis_scripts/results/20241103_create_images_folders/imgs_grids"
	for key_img in key_imgs:
	    f_stem = f"grid_{int(key_img):03d}.png"
	    f_src = f"{dir_img_grids}/{f_stem}"
	    f_tgt = f"{dir_results_person}/{f_stem}"
	    shutil.copy(f_src, f_tgt)



