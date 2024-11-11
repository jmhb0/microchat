"""
python -m ipdb benchmark/graduate_samples/prepare_formdata.py

Load the source csv, choose the `key_question`s, log them, Put them into a new csv
"""
import ipdb
import sys
import json
from pathlib import Path
import pandas as pd
import ast
import os
from PIL import Image
import shutil

from benchmark.refine_bot.run_experiments import _download_csv
from benchmark.refine_bot.run_experiments import get_df_from_key as get_df_from_key_stage1
from benchmark.refine_bot.bot import _stringify_mcq_for_logging
from benchmark.refine_bot.run_eval import _get_filenames_from_key

dir_results_parent = Path(__file__).parent / "results" / Path(__file__).stem
dir_results_parent.mkdir(exist_ok=True, parents=True)

lookup_dfs = {
    'nov5_dspy_full__eval_seed0':
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTjs_xqcIYnxcbIx_BUqpkcJS_D5eBWOs7yipC-WEXsN1PGkc-krImgaRZVdCuVKJKTFzL0snAVz39b/pub?gid=2369810&single=true&output=csv"
}
with open("benchmark/data/formdata_0/2_lookup_question_to_person.json",
          'r') as fp:
    lookup_question_to_person = json.load(fp)
cache_images = {}

df_people = pd.read_csv("benchmark/data/formdata_0/2_df_people.csv")
lookup_person = dict(zip(df_people['key_person'], df_people['Your name']))
# people who dropped out
skip_key_person = [1, 2, 3, 5, 11, 16, 19]


def get_df_from_key(key, overwrite=False):
    """ 
    It also creates the results folder using the same naming convention 
    """
    url = lookup_dfs[key]
    dir_csvs = Path(__file__).parent / "results/"
    dir_csvs.mkdir(exist_ok=True)
    f_csv = dir_csvs / f"IN_{key}.csv"
    f_csv_out = dir_csvs / f"OUT_{key}.csv"
    dir_people_out = dir_csvs / f"OUT_{key}"
    dir_people_out.mkdir(exist_ok=True)
    _download_csv(url, f_csv, overwrite=overwrite)
    df = pd.read_csv(f_csv)
    df = df.set_index('key_question')

    return df, f_csv_out, dir_people_out


def nov5_choose():
    """ """
    df_all, f_csv_out, dir_people_out = get_df_from_key(
        key='nov5_dspy_full__eval_seed0')

    # apply criteria
    df = df_all[df_all["wrong_3_times"] == 1].copy()
    df = df[df['code'].isin(["SUCCESS_NO_CHANGE", "SUCCESS_REWRITE"])].copy()

    # save
    df.to_csv(f_csv_out)

    # save the per-person stuff
    if 0:
        create_form_datasets_perperson(df, dir_people_out)

    return df


def nov9_choose_iter2():
    """ 
    First round of choosing uses the GPT-4o stuff.  
    Unlike the last couple, lets pull in all the results csvs together
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "exp_1109_dspy_full_after_round1_best5_evalseed1_model_gpt-4o-2024-08-06.csv",
        "exp_1109_dspy_full_after_round1_best5_evalseed2_model_gpt-4o-2024-08-06.csv",
        "exp_1109_dspy_full_after_round1_best5_evalseed1_model_anthropicclaude-35-sonnet.csv",
        "exp_1109_dspy_full_after_round1_best5_evalseed2_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)

    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 0]

    f_csv = "benchmark/graduate_samples/results/OUT_nov9_choose_iter2.csv"
    df.to_csv(f_csv)

    return df
    

def nov9_choose_iter3():
    """ 
    First round of choosing uses the GPT-4o stuff.  
    Unlike the last couple, lets pull in all the results csvs together
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "exp_1109_dspy_full_after_round2_o1mini_0_evalseed0_model_gpt-4o-2024-08-06.csv",
        "exp_1109_dspy_full_after_round2_o1mini_0_evalseed1_model_gpt-4o-2024-08-06.csv",
        "exp_1109_dspy_full_after_round2_o1mini_0_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "exp_1109_dspy_full_after_round2_o1mini_0_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)

    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 0]

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov9_choose_iter3.csv"
    df.to_csv(f_csv_out)

    return df

    
def nov9_choose_iter4():
    """ 
    First round of choosing uses the GPT-4o stuff.  
    Unlike the last couple, lets pull in all the results csvs together
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "exp_1109_dspy_full_after_round3_o1mini_best5_evalseed0_model_gpt-4o-2024-08-06.csv",
        "exp_1109_dspy_full_after_round3_o1mini_best5_evalseed1_model_gpt-4o-2024-08-06.csv",
        "exp_1109_dspy_full_after_round3_o1mini_best5_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "exp_1109_dspy_full_after_round3_o1mini_best5_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 0]

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov9_choose_iter4.csv"
    df.to_csv(f_csv_out)

    return df


def nov9_choose_iter5():
    """ 
    First round of choosing uses the GPT-4o stuff.  
    Unlike the last couple, lets pull in all the results csvs together
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "exp_1109_dspy_full_after_round4_o1_0_evalseed0_model_gpt-4o-2024-08-06.csv",
        "exp_1109_dspy_full_after_round4_o1_0_evalseed1_model_gpt-4o-2024-08-06.csv",
        "exp_1109_dspy_full_after_round4_o1_0_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "exp_1109_dspy_full_after_round4_o1_0_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    df = df[df['sum'] == 1]

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov9_choose_iter5.csv"
    df.to_csv(f_csv_out)

    return df


def nov9_choose_iter6():
    """ 
    First round of choosing uses the GPT-4o stuff.  
    Unlike the last couple, lets pull in all the results csvs together
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "exp_1109_dspy_full_after_round1_best5_evalseed0_model_gpt-4o-2024-08-06.csv",
        "exp_1109_dspy_full_after_round1_best5_evalseed1_model_gpt-4o-2024-08-06.csv",
        "exp_1109_dspy_full_after_round1_best5_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "exp_1109_dspy_full_after_round1_best5_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 0]

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov9_choose_iter6.csv"
    df.to_csv(f_csv_out)

    return df


def nov9_choose_iter7():
    """ 
    The one where we collected all the random seeds across all of them
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "all_iters_7_evalseed0_model_gpt-4o-2024-08-06.csv",
        "all_iters_7_evalseed1_model_gpt-4o-2024-08-06.csv",
        "all_iters_7_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "all_iters_7_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 0]
    df = df.groupby('key_question', as_index=False).first()

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov9_choose_iter7.csv"
    df.to_csv(f_csv_out)

    return df

def nov9_choose_iter8():
    """ 
    The one where we collected all the random seeds across all of them
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "all_iters_8_evalseed0_model_gpt-4o-2024-08-06.csv",
        "all_iters_8_evalseed1_model_gpt-4o-2024-08-06.csv",
        "all_iters_8_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "all_iters_8_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 1]
    df = df.groupby('key_question', as_index=False).first()

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov9_choose_iter8.csv"
    df.to_csv(f_csv_out)

    return df

def nov9_choose_iter9():
    """ 
    The one where we collected all the random seeds across all of them
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "all_iters_9_evalseed0_model_gpt-4o-2024-08-06.csv",
        "all_iters_9_evalseed1_model_gpt-4o-2024-08-06.csv",
        "all_iters_9_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "all_iters_9_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 2]
    df = df.groupby('key_question', as_index=False).first()

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov9_choose_iter9.csv"

    return df

def nov10_redoiter1_choose_iter1():
    """ 
    The one where we collected all the random seeds across all of them
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "exp_1110_redo_4o_fromiter10_evalseed0_model_gpt-4o-2024-08-06.csv",
        "exp_1110_redo_4o_fromiter10_evalseed1_model_gpt-4o-2024-08-06.csv",
        "exp_1110_redo_4o_fromiter10_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "exp_1110_redo_4o_fromiter10_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 0]
    df = df.groupby('key_question', as_index=False).first()

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov10_redo1_choose_iter1.csv"
    df.to_csv(f_csv_out)

    return df

def nov10_redoiter1_choose_iter2():
    """ 
    The one where we collected all the random seeds across all of them
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "exp_1110_redo_4o_fromiter1_iter2_evalseed0_model_gpt-4o-2024-08-06.csv",
        "exp_1110_redo_4o_fromiter1_iter2_evalseed1_model_gpt-4o-2024-08-06.csv",
        "exp_1110_redo_4o_fromiter1_iter2_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "exp_1110_redo_4o_fromiter1_iter2_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 0]
    df = df.groupby('key_question', as_index=False).first()

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov10_redo1_choose_iter2.csv"
    df.to_csv(f_csv_out)

    return df

def nov10_redoiter1_choose_iter3():
    """ 
    The one where we collected all the random seeds across all of them
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "exp_1110_redo_4o_fromiter1_iter3_evalseed0_model_gpt-4o-2024-08-06.csv",
        "exp_1110_redo_4o_fromiter1_iter3_evalseed1_model_gpt-4o-2024-08-06.csv",
        "exp_1110_redo_4o_fromiter1_iter3_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "exp_1110_redo_4o_fromiter1_iter3_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 0]
    df = df.groupby('key_question', as_index=False).first()

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov10_redo1_choose_iter4.csv"
    df.to_csv(f_csv_out)

    return df

def nov10_redoiter1_choose_iter4():
    """ 
    The one where we collected all the random seeds across all of them
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "all_iters_redo_4_evalseed0_model_gpt-4o-2024-08-06.csv",
        "all_iters_redo_4_evalseed1_model_gpt-4o-2024-08-06.csv",
        "all_iters_redo_4_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "all_iters_redo_4_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    df = df[df['sum'] == 0]
    df = df.groupby('key_question', as_index=False).first()

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov10_redo1_choose_iter4.csv"
    df.to_csv(f_csv_out)

    return df

def nov10_redoiter1_choose_iter5():
    """ 
    The one where we collected all the random seeds across all of them
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "all_iters_redo_5_evalseed0_model_gpt-4o-2024-08-06.csv",
        "all_iters_redo_5_evalseed1_model_gpt-4o-2024-08-06.csv",
        "all_iters_redo_5_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "all_iters_redo_5_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    
    df = df[df['sum'] == 1]
    df = df.groupby('key_question', as_index=False).first()

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov10_redo1_choose_iter5.csv"
    df.to_csv(f_csv_out)

    return df

def nov10_redoiter1_choose_iter6():
    """ 
    The one where we collected all the random seeds across all of them
    """
    # df_all, f_csv_out, dir_people_out = get_df_from_key(key='nov5_dspy_full__eval_seed0')
    f_evals = [
        "all_iters_redo_5_evalseed0_model_gpt-4o-2024-08-06.csv",
        "all_iters_redo_5_evalseed1_model_gpt-4o-2024-08-06.csv",
        "all_iters_redo_5_evalseed0_model_anthropicclaude-35-sonnet.csv",
        "all_iters_redo_5_evalseed1_model_anthropicclaude-35-sonnet.csv",
    ]
    dir_csvs = "benchmark/refine_bot/results/eval"
    dfs = []
    for f in f_evals: 
        dfs.append(pd.read_csv(os.path.join(dir_csvs, f)))
    df = dfs[0]
    df['r0'] = dfs[0]['pred_correct']
    df['r1'] = dfs[1]['pred_correct']
    df['r2'] = dfs[2]['pred_correct']
    df['r3'] = dfs[3]['pred_correct']
    df['all'] = ((df['r0']==0) & (df['r1']==0) & (df['r2']==0) & (df['r3']==0))
    df['sum'] = df[['r0','r1','r2','r3']].sum(axis=1)
    # apply this strict critera ... can loosen a bit later
    
    df = df[df['sum'] == 2]
    df = df.groupby('key_question', as_index=False).first()

    f_csv_out = "benchmark/graduate_samples/results/OUT_nov10_redo1_choose_iter6.csv"
    df.to_csv(f_csv_out)

    return df


def get_people_whole_round2():
    dir_people_out = dir_results_parent.parent / "OUT_nov9_everything"
    dir_people_out.mkdir(exist_ok=True)
    # save the per-person stuff
    # create_form_datasetms_perperson(df, dir_people_out)

    df_all = get_df_from_key_stage1("dspy_o1-mini_CoTRAG_FULL_nov5", overwrite=True)
    df_all = df_all.set_index("key_question", drop=False)

    # 
    df_completed = nov5_choose()
    df_completed['key_question'] = df_completed.index

    # 
    df2 = nov9_choose_iter2()
    df3 = nov9_choose_iter3()
    df4 = nov9_choose_iter4()
    df5 = nov9_choose_iter5()
    df6 = nov9_choose_iter6()
    df7 = nov9_choose_iter7()
    df8 = nov9_choose_iter8()
    df9 = nov9_choose_iter9()
    df = pd.concat([df2,df3,df4,df5,df6,df7,df8,df9])
    df = df.set_index('key_question', drop=False)

    idxs_remaining = set(df_all.index) - set(df_completed.index)

    # rename the column for df_all
    df_all = df_all.loc[list(idxs_remaining)]
    df_all['choices_postbot'] = df_all['choices']
    df_all['question_postbot'] = df_all['question']

    for key_question, row in df.iterrows():
        df_all.loc[key_question, 'choices_postbot'] = row['choices_postbot']
        df_all.loc[key_question, 'question_postbot'] = row['question_postbot']

    # we're good
    df_all['question_key'] = df_all['key_question']
    create_form_datasets_perperson(df_all, dir_people_out)
    ipdb.set_trace()
    pass


def create_form_datasets_perperson(df, dir_people_out):
    # people lookup
    df.loc[:, 'key_person'] = [
        lookup_question_to_person[str(k)] for k in df['question_key']
    ]

    df = pd.merge(df, df_people, on='key_person').copy()

    for key_person in df['key_person'].unique():
        if key_person in skip_key_person:
            continue

        # person directory
        person_name = lookup_person[key_person]
        dir_results_person = dir_people_out / f"p{key_person}_{person_name.replace(' ' ,'_')}"
        dir_results_person.mkdir(exist_ok=True)

        df_subset = df[df['key_person'] == key_person].copy()

        mcq_strs = []
        for i, row in df_subset.iterrows():
            choices = ast.literal_eval(row["choices_postbot"])
            mcq_str = _stringify_mcq_for_logging(row["question_postbot"],
                                                 choices["choices"],
                                                 choices["correct_index"])
            correct_answer = choices["choices"][choices["correct_index"]]
            mcq_str += f"\n\nCorrect answer: {correct_answer}"
            mcq_strs.append(mcq_str)

        df_subset['mcq_str'] = mcq_strs

        cols = [
            'question_key', 'key_image', 'mcq_str',
            'description_question_answer'
        ]

        df_save = df_subset[cols]
        print(f"{len(df_subset)}\t", person_name)

        key_imgs = df_subset['key_image'].values
        dir_img_grids = "analysis_scripts/results/20241103_create_images_folders/imgs_grids"
        for key_img in key_imgs:
            f_stem = f"grid_{int(key_img):03d}.png"
            f_src = f"{dir_img_grids}/{f_stem}"
            f_tgt = f"{dir_results_person}/{f_stem}"
            shutil.copy(f_src, f_tgt)

        df_save.to_csv(f"{dir_results_person}/data.csv")


if __name__ == "__main__":
    # nov5_choose()
    # nov9_choose_iter2()
    # nov9_choose_iter3()
    # nov9_choose_iter4()
    # nov9_choose_iter5()
    # nov9_choose_iter6()
    # nov9_choose_iter7()
    # nov9_choose_iter8()
    # nov9_choose_iter9()
    # get_people_whole_round2()
    # nov10_redoiter1_choose_iter1()
    # nov10_redoiter1_choose_iter2()
    # nov10_redoiter1_choose_iter3()
    # nov10_redoiter1_choose_iter5()
    nov10_redoiter1_choose_iter6()
    ipdb.set_trace()
    pass
