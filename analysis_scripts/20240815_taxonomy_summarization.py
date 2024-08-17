analysis_scripts/20240815_taxonomy_summarization.py """
	python -m ipdb analysis_scripts/20240815_taxonomy_summarization.py
"""

import ipdb
import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, "..")
sys.path.insert(0, ".")

from models.openai_api import call_gpt

path_resopnses = Path(
    "benchmark/form_process/formdata_0/generated_questions_text/qa_results_keyprompt_1_seed_0_keyprompteval_0.json"
)

prompts = {
    # prompts that try to classify all together - didn't work
    0:
    """\
I am generated a benchmark of VQA quesitons for analysing microscopy images. 
Below is pasted a number of questions that are paired with an image or set of images from a microscopy experiment. 
Please suggest a taxonomy of question types that I could use. 
{QUESTIONS}
""",
    1:
    """\
I am generated a benchmark of VQA quesitons for analysing microscopy images. 
Below is pasted a number of questions that are paired with an image or set of images from a microscopy experiment. 
A few have type '1' or '2' or '3' written before the question. Some also have '-1' written before it, but ignore it. 

Under each type ('1', '2', and '3''), please suggest a taxonomy of question types. 
Specifically, define 3-5 categories that the questions broadly fall under.
{QUESTIONS}
"""
# prompts to classify the types separately 
}
promptkey = 1

# get the questions
with open(path_resopnses) as f:
    y = json.load(f)
qs = []  # all questions
types = []
for k, v in y.items():
    for item in v:
        qs.append(item['question'])
        types.append(item['use_case'])
assert len(qs) == len(types)
print(np.unique(types, return_counts=True))
# when is the use case -1?
idxs_nousecase = np.where(np.array(types) == -1)
np.array(qs)[idxs_nousecase]

# make the prompt
if promptkey == 0:
    prompt = prompts[promptkey]
    questions_str = ""
    for q in qs:
        questions_str += "'" + q + "'" + ",\n"
    prompt = prompt.format(QUESTIONS=questions_str)

    res = call_gpt(prompt, model='gpt-4o-mini', json_mode=False)
    msg = res[0]
    f_save = dir_results / "promptkey1.txt"
    ipdb.set_trace()
   	open(f_save, 'w').write(msg)

elif promptkey == 1:
    prompt = prompts[promptkey]
    questions_str = ""
    for q, type_ in zip(qs, types):
        questions_str += "'" + q + "'" + ",\n"
    prompt = prompt.format(QUESTIONS=questions_str)

    res = call_gpt(prompt, model='gpt-4o-mini', json_mode=False)


ipdb.set_trace()
pass
