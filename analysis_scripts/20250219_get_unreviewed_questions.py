"""
python -m ipdb analysis_scripts/20250219_get_unreviewed_questions.py
"""
import ipdb
from pathlib import Path
import pandas as pd
import json
from benchmark.build_raw_dataset.download_data import download_csv
import numpy as np

from benchmark.graduate_samples.combine_dataset import get_full_dataset_before_review, get_naive_choices_data
df_questions, mcqs = get_full_dataset_before_review()
with open("/Users/jamesburgess/microchat/benchmark/data/formdata_0/2_lookup_question_to_person.json", "r") as f:
    lookup_question_to_person = json.load(f)
df_people = pd.read_csv("/Users/jamesburgess/microchat/benchmark/data/formdata_0/2_df_people.csv")

url_passed1 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5LRmuZWgb20GP6FEGHeB06pnqwV_vuXPK_5CLIl01bYaD8t2SdD5gvIB_dS9SIrmyTAlISEPMkiTZ/pub?gid=278904036&single=true&output=csv"
url_passed2 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5LRmuZWgb20GP6FEGHeB06pnqwV_vuXPK_5CLIl01bYaD8t2SdD5gvIB_dS9SIrmyTAlISEPMkiTZ/pub?gid=1649628245&single=true&output=csv"
url_review1 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGw_NPBcbu_tGelzMaBxPbWI0I5R_7y1VWrEsR2Z-rNhKSFV1FR1UiylwMJ80LwhY9YW-B8bELC42e/pub?gid=0&single=true&output=csv"
url_review2 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGw_NPBcbu_tGelzMaBxPbWI0I5R_7y1VWrEsR2Z-rNhKSFV1FR1UiylwMJ80LwhY9YW-B8bELC42e/pub?gid=1825395971&single=true&output=csv"
dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)

DOWNLOAD = 1
if DOWNLOAD:
    download_csv(url_passed1, dir_results / "passed1.csv")
    download_csv(url_passed2, dir_results / "passed2.csv")
    download_csv(url_review1, dir_results / "review1.csv")
    download_csv(url_review2, dir_results / "review2.csv")

df_passed1 = pd.read_csv(dir_results / "passed1.csv")
df_passed2 = pd.read_csv(dir_results / "passed2.csv")
df_review1 = pd.read_csv(dir_results / "review1.csv")
df_review2 = pd.read_csv(dir_results / "review2.csv")
key_questions = np.concatenate([df_passed1['key_question'], df_passed2['key_question'], df_review1['key_question'], df_review2['key_question']])
df_filtered_questions = df_questions[~df_questions.index.isin(key_questions)].copy()
df_filtered_questions['key_person'] = df_filtered_questions['key_question'].astype(str).map(lookup_question_to_person)
df_filtered_questions['Your name'] = df_filtered_questions['key_person'].map(df_people.set_index('key_person')['Your name'])
print(df_filtered_questions.groupby(['Your name'])['key_question'].count())

df_filtered_questions.to_csv(dir_results / "non-reviewed-questions.csv")
ipdb.set_trace()



ipdb.set_trace()
pass


