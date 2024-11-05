"""
python -m ipdb benchmark/refine_bot/run_experiments.py
"""
import ipdb
from pathlib import Path
import pandas as pd
import requests
import json

from benchmark.refine_bot import bot

### first define some common configs
n_choices_target = 5

model_o1 = "o1-preview-2024-09-12"
model_o1mini = "o1-mini-2024-09-12"
model_gpt4o = "gpt-4o-2024-08-06"
model_gpt4omini = "gpt-4o-mini-2024-07-18"

# yapf: disable
lookup_dfs = {
    "1103_naive_kq3_kc9": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTf4Xzjcosbjdt12M_AyGLP4UimHXZ6uEGK7WDdkAg97ErKuBswkXkmr55CEhMWl3R8FUlEap0AS-1P/pub?gid=1999709323&single=true&output=csv",
}

max_iters = 4
multi_eval = 1
cfg_4o_k0 = dict(
    name="cfg_4o_k0",
    seed=0,
    max_iters=max_iters,
    eval=dict(model=model_gpt4o, key=0, multi_eval=multi_eval),
    reflect=dict(model=model_gpt4o, key=0),
    rewrite=dict(model=model_gpt4o, key=0, strucured_output_key=1, n_choices_target=n_choices_target),
    check_rewrite=dict(model=model_gpt4o, key=0, strucured_output_key=1),
)
# key 1 gpt 4o
cfg_4o_k1 = dict(
    name="cfg_4o_k1",
    seed=0,
    max_iters=max_iters,
    eval=dict(model=model_gpt4o, key=1, multi_eval=multi_eval),
    reflect=dict(model=model_gpt4o, key=1),
    rewrite=dict(model=model_gpt4o, key=1, strucured_output_key=1, n_choices_target=n_choices_target),
    check_rewrite=dict(model=model_gpt4o, key=1, strucured_output_key=1),
)
cfg_o1_k1 = dict(
    name="key1-modelo1",
    seed=1,
    max_iters=max_iters,
    eval=dict(model=model_o1, key=1, multi_eval=multi_eval),
    reflect=dict(model=model_o1, key=1),
    rewrite=dict(model=model_o1, key=1, strucured_output_key=0, n_choices_target=5),
    check_rewrite=dict(model=model_o1, key=1, strucured_output_key=0),
)
cfg_o1_k0 = dict(
    name="cfg_o1_k0",
    seed=0,
    max_iters=max_iters,
    eval=dict(model=model_o1, key=0, multi_eval=multi_eval),
    reflect=dict(model=model_o1, key=0),
    rewrite=dict(model=model_o1, key=0, strucured_output_key=0, n_choices_target=n_choices_target),
    check_rewrite=dict(model=model_gpt4o, key=0, strucured_output_key=1),
)
cfg_o1mini_k1 = dict(
    name="key1-modelo1mini",
    seed=0,
    max_iters=max_iters,
    eval=dict(model=model_o1mini, key=1, multi_eval=multi_eval),
    reflect=dict(model=model_o1mini, key=1),
    rewrite=dict(model=model_o1mini, key=1, strucured_output_key=0, n_choices_target=5),
    check_rewrite=dict(model=model_o1mini, key=1, strucured_output_key=0),
)
# yapf: enable


def get_df_from_key(key, overwrite=False):
    url = lookup_dfs[key]

    dir_csvs = Path(__file__).parent / "results/dfs"
    dir_csvs.mkdir(exist_ok=True)
    f_csv = dir_csvs / f"{key}.csv"
    _download_csv(url, f_csv, overwrite=overwrite)
    df = pd.read_csv(f_csv)

    return df

def _download_csv(url, output_path, overwrite=False):
    if overwrite or not Path(output_path).exists():
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print("CSV downloaded and saved at:", output_path)
    else:
        print("Retrieved cached CSV from:", output_path)


def _get_idxs_sample():
    f_idxs_sample = "benchmark/refine_bot/idxs_sample.json"

    with open(f_idxs_sample, 'r') as fp:
        idxs = json.load(fp)

    return idxs


def exp_1102_first150():
    """
    The first 150 as a list 
    """
    name = "exp_1102_first150"
    df, do_shuffle = _get_data_october('qkey3_ckey9')
    cfg = cfg_4o_k1
    df = df.iloc[:150]

    return df, cfg, name


def exp_1103_test150(seed):
    """ The random sample 150 """
    # configs
    name = f"exp_1103_test150_seed_{seed}"
    cfg = cfg_4o_k1
    cfg['seed'] = seed

    # get dataset 
    df = get_df_from_key("1103_naive_kq3_kc9", overwrite=False)
    idxs = _get_idxs_sample()
    df = df.loc[idxs]

    return df, cfg, name

def exp_1103_test150_o1mini(seed):
    df, cfg, name = exp_1103_test150(seed)
    name="cfg_4o_k1"

    for k in ('eval','reflect', 'rewrite','check_rewrite'):
        cfg[k]['model'] = model_gpt4omini
    name += "_o1mini"

    return df, cfg, name

def exp_1103_test150_multieval_150(seed, multi_eval=3):
    """ 
    The random sample 150. 
    """
    # configs
    name = f"exp_1103_test150multieval{multi_eval}_seed_{seed}"
    cfg = cfg_4o_k1
    cfg['seed'] = seed
    cfg['eval']['multi_eval'] = multi_eval

    # get dataset 
    df = get_df_from_key("1103_naive_kq3_kc9", overwrite=False)
    idxs = _get_idxs_sample()
    df = df.loc[idxs]

    return df, cfg, name



def _get_data_october(questions_source):
    """ 
    These were the data sources from earlier experiments
    """
    if questions_source == 'jeffs_revised':
        url_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuDqK65cBcb0e5y-_DqK5HbFC3raPMP2isPBzTe8tg6vsfTl-7WkDI7NnKTzHJWQ/pub?gid=1746076346&single=true&output=csv"
        f_csv = dir_results_parent / "jeffs_choices.csv"
        if not f_csv.exists():
            download_csv(url_csv, f_csv)
        df = pd.read_csv(f_csv)
        do_shuffle = True

    elif questions_source == 'qkey3_ckey9':
        f_csv = "benchmark/data/formdata_0/question_strategy_3/df_questions_key_choices_9.csv"
        df = pd.read_csv(f_csv)
        shuffle = False  # already shuffle

    else:
        raise ValueError()

    return df, shuffle


if __name__ == "__main__":
    dir_results_parent = Path(__file__).parent / "results" / Path(
        __file__).stem
    dir_results_parent.mkdir(exist_ok=True, parents=True)
    do_multiprocessing = False
    do_multiprocessing = True

    ## run this experiment
    # df, cfg, name = exp_1103_test150(seed=0)
    # df, cfg, name = exp_1103_test150(seed=1)
    # df, cfg, name = exp_1103_test150(seed=2)
    # df, cfg, name = exp_1103_test150(seed=3)
    # df, cfg, name = exp_1103_test150(seed=4)
    # df, cfg, name = exp_1103_test150_o1mini(seed=0)
    df, cfg, name = exp_1103_test150_multieval_150(seed=0, multi_eval=3)

    dir_results = dir_results_parent / f"{name}"
    dir_results.mkdir(exist_ok=True, parents=True)

    bot.run(df,
            cfg,
            dir_results,
            do_multiprocessing=do_multiprocessing,
            do_shuffle=False)
    ipdb.set_trace()
    pass
