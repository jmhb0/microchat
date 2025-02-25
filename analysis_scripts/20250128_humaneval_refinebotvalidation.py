"""
python -m ipdb analysis_scripts/20250128_humaneval_refinebotvalidation.py
# get a set of 50 of Jan's questions, put them in order

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
df_ = df_questions[df_questions.person_name == "Jan Niklas Hansen"]
df = df_.sample(n=50, random_state=0).sort_index()
df['original_question_an_answer'] = [
    f"Question:\n{row['question_0']}\n\nAnswer:\n{row['answer_0']}"
    for i, row in df.iterrows()
]
mcqs = []
letters=list('abcdefg')
for i, row in df.iterrows():
	mcq = row['question'] + "\n\n"
	choices_ = ast.literal_eval(row['choices'])
	choices = choices_['choices']
	correct_index = choices_['correct_index']

	for j, (letter, choice) in enumerate(zip(letters, choices)):

		if j==correct_index:
			letter_str = f"**({letter})**"
		else: 
			letter_str = f"  ({letter})  "

		mcq += f"{letter_str} {choice}\n"
	mcqs.append(mcq)
	
df['final_mcq'] = mcqs
ipdb.set_trace()

df.to_csv(dir_results_parent / "data.csv")
key_imgs = df['key_image'].values
dir_img_grids = "analysis_scripts/results/20241103_create_images_folders/imgs_grids"
for key_img in key_imgs:
    f_stem = f"grid_{int(key_img):03d}.png"
    f_src = f"{dir_img_grids}/{f_stem}"
    f_tgt = f"{dir_results_parent}/{f_stem}"
    shutil.copy(f_src, f_tgt)

ipdb.set_trace()
pass
quit