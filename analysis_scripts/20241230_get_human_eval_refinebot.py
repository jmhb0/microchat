"""
python -m ipdb analysis_scripts/20241230_get_human_eval_refinebot.py
"""
import ipdb
from pathlib import Path
import pandas as pd
import numpy as np

# format: the csv for the data, then the csv for the form response
url_data = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQMBSUoOWhDbnX7Xg17uMKlQrbb0Vyn9lEkV8PRQKuQG2zscXJhs3lN1JxMUwOrI96uOyX7JgbJTdoc/pub?gid=1521923008&single=true&output=csv"
url_responses = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSgmzNNHtSVQjtcW7lXAXoo0j4kqWqppEOt7LDedirvW5iOM1e8yWzahiQzDV1bmsUTXjRVnDfLirrs/pub?gid=1061858550&single=true&output=csv" 

from benchmark.build_raw_dataset.download_data import download_csv

DOWNLOAD = 0
# call: download_csv(url, output_path)
dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)

fname_responses = dir_results / f"responses.csv"
fname_data = dir_results / f"data.csv"
if DOWNLOAD:
    download_csv(url_responses, fname_responses)
    download_csv(url_data, fname_data)

df_responses_ = pd.read_csv(fname_responses)
df_data = pd.read_csv(fname_data)
email = df_responses_.iloc[0].iloc[1]
timestamp = df_responses_.iloc[0].iloc[1]
n_samples = (df_responses_.shape[1] - 2)//3

columns = ['difficulty', 'uses_image', 'trick']
df_responses = pd.DataFrame(np.array(df_responses_)[0, 2:].reshape(
        n_samples, 3), columns=columns)
for c in columns:
    print(c)
    df_responses[c] = [d[0].lower() for d in df_responses[c]]
    print(df_responses.groupby(c)['trick'].count() / len(df_responses))
    print()

ipdb.set_trace()
pass

def get_data_and_responses(fname_data, fname_responses, set_name):
    df_responses_ = pd.read_csv(fname_responses)
    df_data = pd.read_csv(fname_data)

    # put responses array in the right format
    email = df_responses_.iloc[0].iloc[1]
    timestamp = df_responses_.iloc[0].iloc[1]
    n_samples = (df_responses_.shape[1] - 2)
    df_responses = pd.DataFrame(np.array(df_responses_)[0, 2:].reshape(
        n_samples, 1),
                                columns=['answer_letter'])

    df_responses['answer_letter'] = df_responses['answer_letter'].str.lower()
    letters = list("abcdef")
    letter_to_index = dict(zip(letters, range(len(letters))))
    df_responses['answer_index'] = [
        letter_to_index[l[0]] for l in df_responses['answer_letter']
    ]

    df_responses.insert(0, 'email', email)
    df_responses.insert(0, 'idx', idx)
    df_responses.insert(0, 'set', set_name)

    if len(df_data) != len(df_responses):
        raise ValueError("data and feedback not aligned")

    return df_data, df_responses


dfs_responses = []
dfs_data = []
for idx, name in zip(links_humaneval.keys(), names):
    # get the form responses
    fname_responses = dir_results / f"responses_{idx}.csv"
    fname_data = dir_results / f"data_{idx}.csv"
    df_data, df_responses = get_data_and_responses(fname_data,
                                                   fname_responses,
                                                   set_name='nov10')
    df_data['name'] = name

    dfs_responses.append(df_responses)
    dfs_data.append(df_data)
df_responses = pd.concat(dfs_responses)
df_data = pd.concat(dfs_data)
assert len(df_responses) == len(df_data)
df_data['correct'] = (df_data['correct_index_2'] == df_responses['answer_index'])
ipdb.set_trace()
pass




