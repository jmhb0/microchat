"""
python -m ipdb analysis_scripts/20250221_compile_dataset.py
"""
import ipdb
import os
from pathlib import Path
import datasets
import pandas as pd
import json
from benchmark.build_raw_dataset.download_data import download_csv
import numpy as np
import ast
from datasets import Dataset, DatasetDict
from PIL import Image
from datasets.features import Image as ImageFeature
import PIL
import io



from benchmark.graduate_samples.combine_dataset import get_full_dataset_before_review, get_naive_choices_data
df_questions_before_review, mcqs = get_full_dataset_before_review()
def reformat_q_before_review(df_questions_before_review):
    df_questions_before_review['correct_index'] = df_questions_before_review['choices'].apply(lambda x: x['correct_index'])
    df_questions_before_review['choices'] = df_questions_before_review['choices'].apply(lambda x: x['choices'])
    cols = df_questions_before_review.columns.tolist()
    cols.remove('correct_index')
    insert_pos = cols.index('choices') + 1
    cols.insert(insert_pos, 'correct_index')
    df_questions_before_review = df_questions_before_review[cols]
    return df_questions_before_review
df_questions_before_review = reformat_q_before_review(df_questions_before_review)

with open("/Users/jamesburgess/microchat/benchmark/data/formdata_0/2_lookup_question_to_person.json", "r") as f:
    lookup_question_to_person = json.load(f)
df_people = pd.read_csv("/Users/jamesburgess/microchat/benchmark/data/formdata_0/2_df_people.csv")
df_questions_src = pd.read_csv("/Users/jamesburgess/microchat/benchmark/data/formdata_0/4_questions.csv")
df_images_src = pd.read_csv("/Users/jamesburgess/microchat/benchmark/data/formdata_0/4_images.csv").drop(index=0)


url_passed1 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5LRmuZWgb20GP6FEGHeB06pnqwV_vuXPK_5CLIl01bYaD8t2SdD5gvIB_dS9SIrmyTAlISEPMkiTZ/pub?gid=278904036&single=true&output=csv"
url_passed2 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5LRmuZWgb20GP6FEGHeB06pnqwV_vuXPK_5CLIl01bYaD8t2SdD5gvIB_dS9SIrmyTAlISEPMkiTZ/pub?gid=1649628245&single=true&output=csv"
url_review1 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGw_NPBcbu_tGelzMaBxPbWI0I5R_7y1VWrEsR2Z-rNhKSFV1FR1UiylwMJ80LwhY9YW-B8bELC42e/pub?gid=0&single=true&output=csv"
url_review2 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGw_NPBcbu_tGelzMaBxPbWI0I5R_7y1VWrEsR2Z-rNhKSFV1FR1UiylwMJ80LwhY9YW-B8bELC42e/pub?gid=1825395971&single=true&output=csv"
url_review3 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGw_NPBcbu_tGelzMaBxPbWI0I5R_7y1VWrEsR2Z-rNhKSFV1FR1UiylwMJ80LwhY9YW-B8bELC42e/pub?gid=1224242725&single=true&output=csv"
dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)


### get the questions 
DOWNLOAD = 1
if DOWNLOAD:
    download_csv(url_passed1, dir_results / "passed1.csv")
    download_csv(url_passed2, dir_results / "passed2.csv")
    download_csv(url_review1, dir_results / "review1.csv")
    download_csv(url_review2, dir_results / "review2.csv")
    download_csv(url_review3, dir_results / "review3.csv")
df_passed1 = pd.read_csv(dir_results / "passed1.csv")
df_passed2 = pd.read_csv(dir_results / "passed2.csv")
df_review1 = pd.read_csv(dir_results / "review1.csv")
df_review2 = pd.read_csv(dir_results / "review2.csv")
df_review3 = pd.read_csv(dir_results / "review3.csv")


# for the 'passed' reviews:
# set 'question' and 'choices' from 'question_2' and 'choices_2', 
# set 'question_3' and 'choices_3' as the same as question_2 and choices_2
df_passed1 = df_questions_before_review.loc[df_passed1['key_question'].tolist()]
df_passed2 = df_questions_before_review.loc[df_passed2['key_question'].tolist()]

for df in (df_passed1, df_passed2):
    # df['question'] = df['question_2']
    # df['choices'] = df['choices_2']
    df['question_3'] = df['question_2']
    df['choices_3'] = df['choices_2']
    df['correct_index_3'] = df['correct_index_2']

# for the 'review' reviews:
# iterate over the columns in 'revised_mcq_str'. If empty, then add its key_question to the list 'key_questions_skipped'
# call the func 'get_mcq_from_str' which returns string 'question' and dict 'choices'. Assign it to 'question' and 'choices' and also to question_3 and choices_3   
def get_mcq_from_str(text):
    question = text.split('(a)')[0].strip()
    if question[-2:] == "**":
        question = question[:-2]
    question = question.strip()

    num_correct_index = 0
    choices = []
    for idx, letter in enumerate(list('abcde')):
        lines = text.split(f'({letter})')
        if len(lines) < 2:
            assert letter=='e'
        else:
            lines = lines[1]
            if lines[:2] == '**':
                correct_index = idx
                num_correct_index += 1
                lines = lines[2:]
            line = lines.split("\n")[0].strip()
            choices.append(line)
    
    assert num_correct_index == 1, correct_index
    return {
        'question': question,
        # 'choices': dict(choices=choices, correct_index=correct_index)
        'choices': choices, 
        'correct_index': correct_index
    }

key_questions_skipped  = []
key_questions_reviewed = []

for df in (df_review1, df_review2, df_review3):
    df['question'] = None
    df['choices'] = None
    df['question_3'] = None
    df['choices_3'] = None
    for i, row in df.iterrows():
        if row['revised_mcq_str'] == '' or pd.isna(row['revised_mcq_str']):
            key_questions_skipped.append(row['key_question'])
            continue
        key_questions_reviewed.append(row['key_question'])
        ret = get_mcq_from_str(row['revised_mcq_str'])
        df.at[i, 'question'] = ret['question']
        df.at[i, 'choices'] = ret['choices']
        df.at[i, 'correct_index'] = ret['correct_index']
        df.at[i, 'question_3'] = ret['question']
        df.at[i, 'choices_3'] = ret['choices']
        df.at[i, 'correct_index_3'] = ret['correct_index']
print(f"Rows for review: reviewed ", len(key_questions_reviewed))

df = pd.concat([df_passed1, df_passed2, df_review1, df_review2, df_review3])


df = df.set_index('key_question')
print(f"Total data size before filtering unreviewed: {len(df)}")
print(f"Total skipped questions: {len(key_questions_skipped)}")
print(f"Total reviewed questions: {len(key_questions_reviewed)}")
idxs_keep = set(df.index) - set(key_questions_skipped)
assert len(idxs_keep) == len(df) - len(key_questions_skipped)
df = df.loc[list(idxs_keep)]
print("dataset size after filtering unreviewed: ", len(df))

# filter stuff
cols_keep = ['question', 'choices', 'correct_index', 'question_1', 'choices_1',
              'correct_index_1', 'question_2', 'choices_2', 'correct_index_2', 
              'question_3', 'choices_3', 'correct_index_3']
df = df[cols_keep]

# remove the "Question 1:\n" from question_1
df['question_1'] = df['question_1'].str.replace('Question:\n', '')

# get df_questions_src: 
# rename question, answer, comments, and incorrect_answer   to question_0, answer_0, comments_0, and incorrect_answer_0
# rename use_case to task 
# create a col task_str which maps task==1 to 'perception', task==2 to 'hypothesis_gen', and task==3 to 'experimet_proposal'
# filter  df_questions_src to key_question, key_image, key_person, question_0, answer_0, comments_0, task, task_str, incorrect_answer_0
# Process df_questions_src
df_questions_src = df_questions_src.rename(columns={
    'question': 'question_0',
    'answer': 'answer_0',
    'comments': 'comments_0',
    'incorrect_answer': 'incorrect_answer_0',
    'use_case': 'task'
})
task_mapping = {
    1: 'perception',
    2: 'hypothesis_gen',
    3: 'experiment_proposal'
}
df_questions_src['task_str'] = df_questions_src['task'].map(task_mapping)
cols_to_keep = [
    'key_question', 'key_image', 
    'question_0', 'answer_0', 'comments_0',
    'task', 'task_str', 'incorrect_answer_0', 
]
df_questions_src = df_questions_src[cols_to_keep]

# join df_questions_src and df on key_question
df = df.merge(df_questions_src, on='key_question', how='left')

# join on df_images_src
cols_to_keep = ['key_image', 'fnames_images', 'Context - image generation', 'Context - motivation', 'Images - source 1', 'Images source 2',  'caption', 'key_person']
df_images_src = df_images_src[cols_to_keep]
# rename cols 
df_images_src = df_images_src.rename(columns={
    'Context - image generation': 'context_image_generation',
    'Context - motivation': 'context_motivation',
    'Images - source 1': 'images_source_1',
    'Images source 2': 'images_source_2',
    'caption': 'image_caption'
})

# Update image source labels
standard_text = "Question writer owns the image and grants permission to distribute them under the CC-BY-SA 4.0 license with attribution"
ownership_mask = df_images_src['images_source_1'].apply(lambda x: isinstance(x, str) and x.startswith("I own them"))
df_images_src.loc[ownership_mask, 'images_source_1'] = standard_text
df_images_src.loc[ownership_mask, 'images_source_2'] = standard_text
df_images_src.drop(columns=['images_source_1'], inplace=True)
df_images_src = df_images_src.rename(columns={
    'images_source_2': 'images_source'
})



# for df_images_src: load the images based on the list of fnames in the col 'fnames_images'
def get_images(df_images_src):
    imgs_all = []
    idxs = []

    def _get_filenames_from_key(key):
        dir_ = f"benchmark/data/formdata_0/images/idx_{key:04d}"
        fs =  [f for f in os.listdir(dir_) if os.path.isfile(os.path.join(dir_, f))]
        fs = [f for f in fs if f!=".DS_Store"]
        fs = sorted(fs)
        return [os.path.join(os.path.join(dir_, f)) for f in fs]
    
    for key_image, row in df_images_src.iterrows():
        if key_image == 0: 
            continue # was a test image.
        
        fs = _get_filenames_from_key(key_image)
        # imgs_pil = [Image.open(f).convert('RGB') for f in fs]
        # try:
        imgs_pil = [Image.open(f).convert('RGB') for f in fs]
        # imgs = [np.array(img) for img in imgs_pil]
        imgs_all.append(imgs_pil)
        idxs.append(key_image)
    return idxs, imgs_all

idxs, imgs_all = get_images(df_images_src)
assert len(idxs) == len(np.unique(idxs))
df_images_src['images'] = [imgs_all[idxs.index(key_image)] for key_image in df_images_src['key_image']]
df_images_src.drop('fnames_images', axis=1, inplace=True)
df = df.merge(df_images_src.set_index('key_image'), on='key_image', how='left')

def df_harmonize_choices_type(df):
    choice_cols = ['choices_1', 'choices_2', 'choices_3']
    
    for col in choice_cols:
        for idx in df.index:
            val = df.at[idx, col]
            if isinstance(val, str):
                try:
                    parsed_val = ast.literal_eval(val)
                    assert isinstance(parsed_val, list), f"Column {col}, index {idx}: Expected list after parsing, got {type(parsed_val)}"
                    df.at[idx, col] = parsed_val
                except Exception as e:
                    raise ValueError(f"Error processing column {col}, index {idx}: {str(e)}")
    return df
# Add this line after the merge operations and before creating the Dataset
df = df_harmonize_choices_type(df)

# Cast correct_index columns to int
correct_index_cols = [col for col in df.columns if col.startswith('correct_index')]
for col in correct_index_cols:
    df[col] = df[col].astype(int)

# Create dataset and add images
images = df['images']
df.drop(columns=['images'], inplace=True)
# Move the columns for '0', which are user-specified to the start
cols_to_move = ['key_image','question_0', 'answer_0', 'comments_0', 'incorrect_answer_0']
new_positions = [1, 5, 6, 7, 8]
df_reordered = df.copy()
for col in cols_to_move:
    df_reordered.drop(columns=[col], inplace=True)
for col, pos in zip(cols_to_move, new_positions):
    df_reordered.insert(pos, col, df[col])
df = df_reordered

correct_answers= df.apply(lambda row: row['choices'][row['correct_index']], axis=1)
df.insert(5, 'correct_answer', correct_answers)

# for testing only
if 0:
    df = df[24:28]
    images = images[24:28]
dataset = Dataset.from_pandas(df)

# Add images list column and set features
def pil_to_bytes(pil_img):
    with io.BytesIO() as output:
        pil_img.save(output, format="PNG")
        return output.getvalue()

images_as_bytes = [
    [pil_to_bytes(img) for img in image_list]
    for image_list in images
]

current_columns = dataset.column_names
dataset = dataset.add_column("images_list", images_as_bytes)
features = dataset.features.copy() if dataset.features is not None else {}
features["images_list"] = datasets.Sequence(datasets.Image())
dataset = dataset.cast(features)
# and reorder the columns
new_column_order = current_columns[:2] + ["images_list"] + current_columns[2:]
dataset = dataset.select_columns(new_column_order)

ipdb.set_trace()
dataset.push_to_hub('jmhb/microvqa')
ipdb.set_trace()
pass
# put `df`
