"""
python -m ipdb analysis_scripts/20251015_test_jeffs_questions.py
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
from models.openai_api import call_gpt_batch
import re

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from benchmark.build_raw_dataset.download_data import download_csv

idxs_question = [136, 137, 138, 139, 140, 142, 145]

model = "gpt-4o-2024-08-06"
dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)
data_dir = Path("benchmark/data/formdata_0")
verbose = 0

PROMPT_SUFFIX = """Think step by step and then output the answer in the format of \"The answer is (X)\" at the end."  """
regex_pattern = r"answer is \(?([a-zA-Z])\)?"

# get the relevant questions
# direct link to the sheet: https://docs.google.com/spreadsheets/d/1KOg4JU6mLIpMTK1Y0BZoosu5zJGqEyxa/edit?gid=1746076346#gid=1746076346
url_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuDqK65cBcb0e5y-_DqK5HbFC3raPMP2isPBzTe8tg6vsfTl-7WkDI7NnKTzHJWQ/pub?gid=1746076346&single=true&output=csv"
f_csv = dir_results / "jeffs_choices.csv"
download_csv(url_csv, f_csv)
df = pd.read_csv(f_csv)
df = df[df['key_question'].isin(idxs_question)]
assert df['multiple_choice'].isna().sum() == 0, "missing multiple choices"
assert df['revised_question'].isna().sum() == 0, "missing multiple choices"

###### MCQ: build the prompt and get the images ######
key_images = set(df['key_image'])
cache_images = {}
batch_prompts_imgs = []
batch_prompts_text = []
batch_prompts_text_open = []
for idx, row in df.iterrows():
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
    prompt_text = row['revised_question'] + "\n\n" + row['multiple_choice']
    prompt_text += "\n" + PROMPT_SUFFIX
    remove_strings = ["(Correct)", "(Incorrect)", "(correct)", "(incorrect)"]
    for s in remove_strings:
        prompt_text = prompt_text.replace(s, "")
    batch_prompts_text.append(prompt_text)

    batch_prompts_text_open.append(row['revised_question'])

# overwrite_cache = True
res = call_gpt_batch(batch_prompts_text,
                     batch_prompts_imgs,
                     overwrite_cache=True,
                     json_mode=False)

# extract mcq 
msgs = [r[0] for r in res]
preds = []
for msg in msgs:
    match = re.search(regex_pattern, msg)
    if match is not None:
        pred = match.group(1)
        preds.append(pred)
    else:
        preds.append(-1)
question_to_answer = dict(zip(idxs_question, preds))
print(f"MCQ preds", question_to_answer)


###### run in open mode ######
res = call_gpt_batch(batch_prompts_text_open,
                     batch_prompts_imgs,
                     overwrite_cache=True,
                     json_mode=False)
msgs_open = [r[0] for r in res]


###### some basic logging ######
for i in range(len(df)):
    str_log = batch_prompts_text[i]
    str_log += "\n\n" + "-" * 80 + "\n\n"
    str_log += msgs[i]
    str_log += "\n\n" + "-" * 80 + "\n\n"
    str_log += msgs_open[i]

    row = df.iloc[i]
    f_questions = dir_results / f"qkey_{row['key_question']}_ikey_{row['key_image']}.txt"
    open(f_questions, "w").write(str_log)
ipdb.set_trace()
pass
