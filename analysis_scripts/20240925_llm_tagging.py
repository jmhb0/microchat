"""
python analysis_scripts/20240925_llm_tagging.py

btw, the main scripts already have some tagging functions for use case 
see script "benchmark/build_raw_dataset/llm_based_annotations_1.py"

- Use case 

- Modality: histopath, light microscopy, fluoro microscopy, EM/ET, single particle 
- Some text summary 

- 2nd level taxonomy 
- Question has the answer? 
- Free form response matches the response?
- Suggest an even lower taxonomy level like “structure-function” etc. 
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

sys.path.insert(0, "..")
sys.path.insert(0, ".")

from models.openai_api import call_gpt_batch

model = "gpt-4o-2024-08-06"

dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)
data_dir = Path("benchmark/data/formdata_0")

# https://platform.openai.com/docs/models/gpt-4o
# more expensive and better
model_4o = "gpt-4o-2024-05-13"
# much cheaper but a little worse, not sure how much worse
model_4omini = "gpt-4o-mini-2024-07-18	"

## raw vqa bench
f_images = data_dir / "4_images.csv"
f_questions = data_dir / "4_questions.csv"
df_images = pd.read_csv(f_images)
df_questions = pd.read_csv(f_questions)

## `df` is the generated choices. samples coming from the first set of generated questions and chocesk
key_question_gen = 0
key_choices_gen = 0
f_questions_choices = data_dir / f"question_strategy_{key_question_gen}" / f"df_questions_key_choices_{key_choices_gen}.csv"
df = pd.read_csv(f_questions_choices)

# some useful strings
df['question_and_answer'] = [
    f"{q}\n\nAnswer:\n```{a}```"
    for q, a in zip(df_questions['question'], df_questions['answer'])
]
df['question_and_answer_and_context'] = [
    f"{q}\n\nAnswer:\n```{a}```"
    for q, a in zip(df_questions['question'], df_questions['answer'])
]

### use case prediction
prompt_template_usecase = """
Below is a text that is paired with a microscopy image. 
The text is some context, question, and answer.

Choose which 'use case' best describes this question. The options are:
1. What is unusual or interesting in these images? And why is it unusual or interesting? It often asks about image features. 
2. What are the possible mechanisms that could cause it? It focuses on underlying causes for image content.
3. What should we do next and why? It often suggests next steps. 
You can pick one option only 

Response in json, one option only, so one of: 
{'use_case' : "1"} or {'use_case' : "2"} or {'use_case' : "3"}

Here is the text
```
{{text}}
```"""
if 1:
	print("Use case generation")
	json_mode = True
	prompts = [
	    prompt_template_usecase.replace("{{text}}", t)
	    for t in df['question_and_answer']
	]
	res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
	cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
	print(f"Cost ${cost:.2f} with model {model}")
	use_cases = [r[0]['use_case'] for r in res]
	df['_use_case_pred'] = use_cases

### eli5: "explain it like I'm 5"
prompt_template_eli5 = """
Below is a text that is paired with a microscopy image. 
The text is some context, question, and answer.

The content requires biological understanding. 
Please simplify the question to more basic English for people who only have basic biological knowledge. 
It's okay if some of the detail is missing - just give the general idea of what the question is asking.

Return json: {"summary" : "..."}

Here is the text to summarize:
```
{{text}}
```"""
if 1:
	json_mode = True
	print("\nDoing eli5 summarisation")
	prompts = [
	    prompt_template_eli5.replace("{{text}}", t)
	    for t in df['question_and_answer']
	]
	res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
	cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
	print(f"Cost ${cost:.2f} with model {model}")
	df['_summary_1_pred'] = [r[0]['summary'] for r in res]

### eli5, like before, except also we have include the image context
prompt_template_eli5 = """
Below is a text that is paired with a microscopy image. 
The text is some context, question, and answer.

The content requires biological understanding. 
Please simplify the question to more basic English for people who only have basic biological knowledge. 
It's okay if some of the detail is missing - just give the general idea of what the question is asking.

Return json: {"summary" : "..."}

Here is the text to summarize:
```
{{text}}
```"""
if 1:
	json_mode = True
	print("\nDoing eli5 summarisation 2")
	prompts = [
	    prompt_template_eli5.replace("{{text}}", t)
	    for t in df['question'] # this field is after image construction
	]
	res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
	cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
	print(f"Cost ${cost:.2f} with model {model}")
	df['_summary_2_pred'] = [r[0]['summary'] for r in res]


### modality and scale. This one queries image and not question, so fewer queries
prompt_template_modality_and_scale = """\
Below is a text string explaining how an image or set of related images were created. 

Based on that text, classify the image based to the microscopy imaging modality. 
One of: 'light_microscopy', 'fluorescence_microscopy', 'electron_microscopy'. Make your best guess.

Then classify into scale. One of: "tissue", "cellular", "subcellular". 
Return json {'modality' : '...', 'scale' : '...'}

TEXT: 
```
{{text}}
```
"""
if 1:
    json_mode = True
    print("\nDoing modality and scale summarization")
    prompts = [
        prompt_template_modality_and_scale.replace("{{text}}", str(t))
        for t in df_images['Context - image generation']
    ]
    res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
    cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
    print(f"Cost ${cost:.2f} with model {model}")
    df_images['modality'] = [r[0]['modality'] for r in res]
    df_images['scale'] = [r[0]['scale'] for r in res]
    assert len(res) == len(df_images)
    df['_image_modality_pred'] = df['key_image'].map(df_images['modality'].to_dict())
    df['_image_scale_pred'] = df['key_image'].map(df_images['scale'].to_dict())

ipdb.set_trace()
pass

###
f_save = dir_results / "df_choices_with_llm_preds.csv"
df.to_csv(f_save)