"""
python -m ipdb analysis_scripts/20241103_choose_subset.py

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
import pickle
import ast
from PIL import Image
from models.openai_api import call_gpt_batch, call_gpt
import re
from pydantic import BaseModel
from omegaconf import OmegaConf
import logging
from datetime import datetime
import glob
import csv

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from benchmark.build_raw_dataset.download_data import download_csv

dir_results_parent = Path(__file__).parent / "results" / Path(__file__).stem
dir_results_parent.mkdir(exist_ok=True, parents=True)

# from this sheet https://docs.google.com/spreadsheets/d/18JbxmDYNw_92Z2RIxVAO1UwCDDb85oLbto_rxfwnXAs/edit?gid=1999709323#gid=1999709323
url_questions = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTf4Xzjcosbjdt12M_AyGLP4UimHXZ6uEGK7WDdkAg97ErKuBswkXkmr55CEhMWl3R8FUlEap0AS-1P/pub?gid=1999709323&single=true&output=csv"
f_csv = dir_results_parent / "df_questions_all.csv"
if not Path(f_csv).exists():
	download_csv(url_questions, f_csv)
df_questions_all = pd.read_csv(f_csv).set_index('key_question')
# df_people = pd.read_csv("benchmark/data/formdata_0/2_df_people.csv")
idxs_all = df_questions_all.index.values

sample_size = 150
idxs = np.random.choice(idxs_all, size=sample_size, replace=False)
df = df_questions_all.loc[idxs]

idxs_lst = sorted([int(idx) for idx in idxs])

with open(dir_results_parent / "idxs_sample.json", "w") as fp:
	json.dump(idxs_lst, fp, indent=4)


# get some metrics to make sure it all makes sense

ipdb.set_trace()
pass 


