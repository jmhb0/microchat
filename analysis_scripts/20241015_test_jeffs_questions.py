"""
python -m ipdb analysis_scripts/20241015_test_jeffs_questions.py
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

seed = 0  # controling the MCQ order
do_shuffle = True

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from benchmark.build_raw_dataset.download_data import download_csv

idxs_question = [136, 137, 138, 139, 140, 142, 145]
idxs_question = [
    136, 137, 138, 139, 140, 142, 145, 176, 177, 178, 179, 180, 181, 187, 188,
    189, 190, 191, 192, 193, 194, 205, 206, 207, 538, 539, 540, 541, 542, 543
]

model = "gpt-4o-2024-08-06"
dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)
data_dir = Path("benchmark/data/formdata_0")
verbose = 0

PROMPT_SUFFIX = """Think step by step and then output the answer in the format of \"The answer is (X)\" at the end."  """
regex_pattern = r"answer is \(?([a-zA-Z])\)?"


def extract_mcs(msgs):
    preds = []
    for msg in msgs:
        match = re.search(regex_pattern, msg)
        if match is not None:
            pred = match.group(1)
            preds.append(pred)
        else:
            preds.append(-1)
    return preds


PROMPT_PREFIX_NO_IMAGE = """The following question is supposed to be paired with an image. We will not provide the image, so answer to the best of your ability."""

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
batch_prompts_text_no_image = []
gts = []  # index of the gt

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

    ### handle the multiple choice
    choices = row['multiple_choice']
    remove_strings = ["(Correct)", "(Incorrect)", "(correct)", "(incorrect)"]
    for s in remove_strings:
        choices = choices.replace(s, "")
    # make it a list
    choices = choices.split("\n")
    assert len(choices) in (4, 5)
    choices = [c[3:] for c in choices]  # remove the marker at the start
    # now shuffle
    idxs = np.arange(len(choices))
    np.random.seed(seed + row.name)  # shuffle seed depends on the row
    if do_shuffle:
        idxs = np.random.permutation(idxs)
    choices = [choices[idx] for idx in idxs]
    # add letters to it
    str_choices = ""
    letters = ['a', 'b', 'c', 'd', 'e']
    for letter, choice in zip(letters, choices):
        str_choices += f"({letter}) {choice}\n"
    # record the correct answer
    idx_correct = int(np.where(idxs == 0)[0][0])
    letter_correct = letters[idx_correct]
    gts.append(letter_correct)

    # save the text prompt
    prompt_text = row['revised_question'] + "\n\n" + str_choices + PROMPT_SUFFIX
    batch_prompts_text.append(prompt_text)

    # no image prompt
    prompt_text_no_image = PROMPT_PREFIX_NO_IMAGE + prompt_text
    batch_prompts_text_no_image.append(prompt_text_no_image)

    # open prompt
    batch_prompts_text_open.append(row['revised_question'])

# overwrite_cache = True
overwrite_cache = False
res = call_gpt_batch(batch_prompts_text,
                     batch_prompts_imgs,
                     overwrite_cache=False,
                     json_mode=False)

# mcq
msgs = [r[0] for r in res]
preds = extract_mcs(msgs)

# mcq no image
overwrite_cache = False
res = call_gpt_batch(batch_prompts_text_no_image,
                     imgs=None,
                     overwrite_cache=overwrite_cache,
                     json_mode=False)
msgs_no_image = [r[0] for r in res]
preds_no_image = extract_mcs(msgs_no_image)

###### run in open mode ######
overwrite_cache = False
res = call_gpt_batch(batch_prompts_text_open,
                     batch_prompts_imgs,
                     overwrite_cache=overwrite_cache,
                     json_mode=False)
msgs_open = [r[0] for r in res]

###### some basic logging ######
for i in range(len(df)):
    row = df.iloc[i]
    f_questions = dir_results / f"qkey_{row['key_question']}_ikey_{row['key_image']}.txt"

    str_log = ""

    str_log += f"MCQ standard question index {idxs_question[i]}, correct option {gts[i]}:"
    str_log += "\n" + "-" * 80
    str_log += "\nPrompt:" + "-" * 80 + "\n"
    str_log += "\n" + batch_prompts_text[i]
    str_log += "\n" + "-" * 80 + "\nResponse:\n"
    str_log += msgs[i]

    str_log += "\n\n" + "-" * 80 + "\n\n"

    str_log += f"MCQ no image question index {idxs_question[i]}, correct option {gts[i]}:"
    str_log += "\n" + "-" * 80
    str_log += "\nPrompt:" + "-" * 80 + "\n"
    str_log += "\n" + batch_prompts_text_no_image[i]
    str_log += "\n" + "-" * 80 + "\nResponse:\n"
    str_log += msgs_no_image[i]

    str_log += "\n\n" + "-" * 80 + "\n\n"

    str_log += f"Open with image {idxs_question[i]}:"
    str_log += "\n" + "-" * 80
    str_log += "\nPrompt:" + "-" * 80 + "\n"
    str_log += "\n" + batch_prompts_text_open[i]
    str_log += "\n" + "-" * 80 + "\nResponse:\n"
    str_log += msgs_open[i]

    open(f_questions, "w").write(str_log)

# summary 
df_res = pd.DataFrame(
    {
        'gt': gts,
        'pred': preds,
        'pred_no_image': preds_no_image
    },
    index=idxs_question)
df_res['1'] = (df_res['gt'] == df_res['pred'])
df_res['1_no_image'] = (df_res['gt'] == df_res['pred_no_image'])

acc = float(df_res['1'].mean())
acc_noimg = float(df_res['1_no_image'].mean())
print("Acc overall", acc)
print("Acc no imag", acc_noimg)
# print(df_res)

ipdb.set_trace()
pass
