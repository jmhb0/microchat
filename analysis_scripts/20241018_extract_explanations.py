"""
python -m ipdb analysis_scripts/20241018_extract_explanations.py
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
import re

sys.path.insert(0, "..")
sys.path.insert(0, ".")

from models.openai_api import call_gpt_batch
dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)

df = pd.read_csv("/Users/jamesburgess/microchat/benchmark/data/formdata_0/question_strategy_0/df_questions_key_choices_3.csv")

prompt_template = """
Below is a question and a prompt related to a microscopy image or set of images. 
And below that is an "answer_short" and "answer_full".
The "answer_short" is derived from the "answer_full", but is most likely shorter and simpler.

Using the information in "answer_full", provide an "explanation" that provides reasoning behind the "answer_short".
Your explanation should avoid injecting new information from your knowledge base. Only use the information available in "answer_full".
Return json 
{"explanation" : "..."}

{{question}}

ANSWER_SHORT:
```
{{answer_short}}
```

ANSWER_FULL:
```
{{answer_full}}
```
"""

batch_prompts_text = []
for i, row in df.iterrows():
	choices = ast.literal_eval(row['choices'])
	answer_short = choices['choices'][choices['correct_index']]

	prompt = prompt_template.replace("{{question}}", row['question'])
	prompt = prompt.replace("{{answer_short}}", answer_short)
	prompt = prompt.replace("{{answer_full}}", str(row['answer']))

	batch_prompts_text.append(prompt)

batch_prompts_text = batch_prompts_text[:200]


overwrite_cache = False
res = call_gpt_batch(batch_prompts_text,
					 imgs=None,
					 overwrite_cache=overwrite_cache,
					 json_mode=True)
cost = sum([r[1] for r in res])
msgs = [m[0]['explanation'] for m in res]




ipdb.set_trace()
pass
