"""
python -m ipdb analysis_scripts/20241230_get_human_eval.py
"""
import ipdb
from pathlib import Path
import pandas as pd
import numpy as np

# format: the csv for the data, then the csv for the form response
names = ['alex','connor','disha','jesus','ridhi', 'will']
links_humaneval = {
    # alex
    0: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTSvXVCHpshVRoJRpJmQovJ9dgw_BxGV9Tu8uuZupbIkQadgCROvvh22I4E-CY3p_IobNSGNx7vxo6w/pub?gid=58362206&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRagAAvdnrszWBAmRKQzZpxYqSGqZXXKyo84NJNaJ68V0F89kP3Dqay0-_s95IIC2F-ryZin_u3chqL/pub?gid=1053452731&single=true&output=csv",
    },
    # connor
    1: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vR6yVsXtbVuoabywly8RFVwhvXXLkkiBYOzPdbhC3_Xj7voV0XA0aFfzBg4bzBR-Y4XnU2RYS3cnEdx/pub?gid=2142367551&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vS8Ri1--BYcFZhvMAyL8avTi2B54ngmYW26Qf-m-4mSe_gD5e3knlLFEL-ngwTnEBXmUi8oavbxHF7k/pub?gid=1166378803&single=true&output=csv",
    },
    # disha
    2: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSJN8F3Z4_SXEl40DuuDrE8py_agJCMrCtCAeFJpIZBskAUwQSWX7ILZicqZol0aXHDYTOnskyPNify/pub?gid=1035321508&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vR080LLCf-c1C0SRb9TtcqUTVxbvRivtRZ9I3YUyXX5arlSCzhLaDoIroQDHTJV3dNiuCHG8kX5bxa_/pub?gid=747064086&single=true&output=csv",
    },
    # jesus
    3: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vS1T-1V-jACZrmOOe9u7K2KR42ass-frk9brq9RU6MZjxg9HdZglXS1eRIvv0e_i7-_EZ2di3hRB9Ox/pub?gid=1849655952&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ_xj21djUooATy5clDfcrX_7_MnFVMkp8ELP5T0_MgqcLnkkVd6IpZJMLabcnOoyjdA4JY-YAoDq43/pub?gid=638484303&single=true&output=csv",
    },
    #  ridhi
    4: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQn5k0LSifSaR8DMRyxL1v85duXjLNS7iSTJR3kCphgu-JO3DJlerPFDPJ5z5pPWabIQROQHqd3Mdw/pub?gid=2066439765&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRSYoPOWvlVuh_iHua6PfFeVePxr4HuU82JUqOsVGFvJ239rv5xsWHaip5ko2msKuhKwuUegr6d3Vr5/pub?gid=762448081&single=true&output=csv",
    },
    5: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSaeLBeWK1hQioNL4wj9NL_c7jfo1s2flIcfAYBrmElvXmDtKXpwb5QXKXcFyfzESdcbMkUJDa8qaOZ/pub?gid=1060255948&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTh-gtzvPbQPTWyQvDeL5X4DHLvnBR7OCj6Pk4Y3s8mlxSP-Oe97h6V04LSMWGhGtZ9IOxGU4yc0Jr_/pub?gid=617831291&single=true&output=csv",
    },
}

from benchmark.build_raw_dataset.download_data import download_csv

DOWNLOAD = 0
# call: download_csv(url, output_path)
dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)

if DOWNLOAD:
    for idx, v in links_humaneval.items():
        print("downloading idx ", idx)

        fname = dir_results / f"responses_{idx}.csv"
        if len(v['response']) > 0:
            download_csv(v['response'], fname)

        fname = dir_results / f"data_{idx}.csv"
        if len(v['data']) > 0:
            download_csv(v['data'], fname)


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




