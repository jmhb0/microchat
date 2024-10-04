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
import ast

sys.path.insert(0, "..")
sys.path.insert(0, ".")

from models.openai_api import call_gpt_batch

# https://platform.openai.com/docs/models/gpt-4o
model = "gpt-4o-2024-08-06"

dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)
data_dir = Path("benchmark/data/formdata_0")

## data
# first get the dataframe with the questions and generated mcq's
key_question_gen = 0
key_choices_gen = 0
f_questions_choices = data_dir / f"question_strategy_{key_question_gen}" / f"df_questions_key_choices_{key_choices_gen}.csv"
df = pd.read_csv(f_questions_choices)
# also the images and questions dataframes
df_images = pd.read_csv(data_dir / "4_images.csv")
df_questions = pd.read_csv(data_dir / "4_questions.csv")

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
    df['_use_case'] = use_cases

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
    df['_summary_1'] = [r[0]['summary'] for r in res]

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
        for t in df['question']  # this field is after image construction
    ]
    res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
    cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
    print(f"Cost ${cost:.2f} with model {model}")
    df['_summary_2'] = [r[0]['summary'] for r in res]

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
    df['_image_modality'] = df['key_image'].map(
        df_images['modality'].to_dict())
    df['_image_scale'] = df['key_image'].map(df_images['scale'].to_dict())

### sub-use-cases
prompt_templates_usecase_lvl2 = {
    '1':
    """\
Below is a text that is paired with a microscopy image. 
The text is some context, question, and answer.
This example has been classified into the following use-case:
	"1. What is unusual or interesting in these images? And why is it unusual or interesting? It often asks about image features."

Your task is to choose which 'sub-use case' best describes this question. The options are:
1.1: Comparing image pairs or image sets, for example:
	 "how is the mitochondrial morphology different in image 1 vs image 2?"
1.2: Identifying abnormalities, for example:
	"Are the nuclei healthy or unhealthy and what features tell you that?"
1.3: Identifying technical issues, for example 
	"we stained for tubulin. Was it successful in terms of localization and SNR?”
	"We see something unusual: is it biologically meaningful, or an experimental artifact?"

Respond in json: {"sub_use_case" : "1.1|1.2|1.3"}

TEXT: 
```
{{text}}
```
""",
    '2':
    """\
Below is a text that is paired with a microscopy image. 
The text is some context, question, and answer.
This example has been classified into the following use-case:
	"2. What are the possible mechanisms that could cause it? It focuses on underlying causes for image content."

Your task is to choose which 'sub-use case' best describes this question. The options are:
2.1: Biological mechanisms - causal, for example
	"What gene dysregulation could lead to the observed cytoplasm shape?"
2.2: Biological implications - functional, for example
	"Given the unexpected localization of the centrosome, what will be the impact on liver function?"	
2.3: Technical explanations, for example:
	"why does increased laser power lead to more release of mScarlet-CD4 from the golgi to the cell membrane?"

You can pick one option only.
Respond in json: {"sub_use_case" : "2.1|2.2|2.3"}

TEXT: 
```
{{text}}
```
	""",
    '3':
    """\
Below is a text that is paired with a microscopy image. 
The text is some context, question, and answer.
This example has been classified into the following use-case:
	"3. What should we do next and why? It often suggests next steps."

Your task is to choose which 'sub-use case' best describes this question. The options are:
3.1: Suggest a new experiment to test some biological hypothesis, for example:
	"What new experiment could I do to test if Gene Y is causing these problems?"
3.2: Suggest an experimental change to address a technical issues, for example
	"The staining didn’t target what I wanted, and the SNR was too low. How can I improve it?"

You can pick one option only.
Respond in json: {"sub_use_case" : "3.1|3.2"}

TEXT: 
```
{{text}}
```
""",
}

if 1:
    print("\nSub-use case generation")
    json_mode = True
    for use_case in ('1', '2', '3'):
        idxs = df[df['_use_case'] == use_case].index
        prompts = [
            prompt_templates_usecase_lvl2[use_case].replace("{{text}}", t)
            for t in df.loc[idxs, 'question_and_answer']
        ]
        res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
        cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
        print(
            f"Cost for use-case {use_case} with ${cost:.2f} with model {model}"
        )
        sub_use_cases = [r[0]['sub_use_case'] for r in res]
        df.loc[idxs, '_sub_use_case'] = sub_use_cases

###
prompt_template_question_has_answer = """
Below is a text that is paired with a microscopy image. 
The text is some context, question, and answer.

Sometimes the question is poorly phrased, so that the answer is already contained in the conext or question. 
Here is an example where the answer is in the context: 
```
Description of image preparation: I stained these cells with a protein that is a marker for the centrosome, and imaged with confocal microscopy. 
Question: What is the protein localization?
Answer: centrosome
```
Here is an example where the answer is in the question
```
Description of image preparation: I stained these cells with a protein and imaged with confocal microscopy.
Question: The centrosome-marker protein is localized to which organelle?
Answer: centrosome
```

Your task is to identify if this occurs in the question, return "1" if it is and "0" otherwise. 
Also give an explanation. 
The json output should be:
{"is_answered" : "0|1, "explanation" : "..."}

Here is the context and question:
```
{{text_question}}
Answer:
{{text_answer}}
```"""
if 1:
    json_mode = True
    print("\nRunning 'question has answer' test")
    prompts = []
    for t_q, t_a in zip(df['question'], df['answer']):
    	prompt = prompt_template_question_has_answer.replace("{{text_question}}", str(t_q))
    	prompt = prompt.replace("{{text_answer}}", str(t_a))
    	prompts.append(prompt)
        
    res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
    cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
    print(f"Cost ${cost:.2f} with model {model}")
    df['_question_has_answer'] = [r[0]['is_answered'] for r in res]
    df['_question_has_answer_explanation'] = [
        r[0]['explanation'] for r in res
    ]

### testing whether the answer in 'choices' matches what was actually provided 
prompt_template_choices_matches_freeform = """
Below is some text that is paired with a microscopy image. 
We show  CONTEXT_AND_QUESTION which is a context and question about the microscopy image.
Then we give an ANSWER which is the answer to the question. 
Then we give an ANSWER_GENERATED which is a rewritten version of the ANSWER.

Your task is to verify whether ANSWER_GENERATED is equivalent or different to ANSWER. 
It's okay if ANSWER_GENERATED is shorter and simplified than ANSWER

Return a json with `is_different` that is 1 if different, or 0 if equivalent. 
Then include an explanation:
{"is_different" : "0|1", "explanation" : "..."}

CONTEXT_AND_QUESTION:
```
{{question_and_context}}
```
ANSWER:
```
{{answer}}
```
GENERATED_ANSWER:
```
{{generated_answer}}
```
"""
if 1:
    json_mode = True
    print("\nRunning test for whether the generated answer in choices matches the freeform answer")

    prompts = []
    all_choices = []
    for t_q, t_a, t_c in zip(df['question'], df['answer'], df['choices']):
    	prompt = prompt_template_choices_matches_freeform
    	prompt = prompt.replace("{{question_and_context}}", str(t_q))
    	prompt = prompt.replace("{{answer}}", str(t_a))
    	choices = ast.literal_eval(t_c)
    	all_choices.append(choices)
    	choices_ans = choices['choices'][choices['correct_index']]
    	prompt = prompt.replace("{{generated_answer}}", choices_ans)
    	prompts.append(prompt)
    	
    res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
    cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
    print(f"Cost ${cost:.2f} with model {model}")
    # {"is_different" : "0|1", "explanation" : "..."}
    df['_answer_generation_is_different'] = [r[0]['is_different'] for r in res]
    df['_answer_generation_is_different_explanation'] = [
        r[0]['explanation'] for r in res
    ]

# test Laura's idea about the correct answer being the longest 
if 1: 
	choices = [ast.literal_eval(d) for d in df['choices']]
	correct_idxs = [c['correct_index'] for c in choices]
	# lengts[i][j] is the string length of the jth choice of sample i
	lengths = [[len(s) for s in lst['choices']]  for lst in choices]
	# longest[i] is the string length of the correct choice of sample i
	longest = [int(np.argmax(l)) for l in lengths]
	# pcnt of samples where the longest string is the correct choice
	is_longest = np.array(correct_idxs) == np.array(longest)
	pcnt_is_longest = is_longest.sum() / len(is_longest)
	print(f"\nPcnt of examples where correct choice is the longest {100*pcnt_is_longest:.0f}%")
	lengths_longest = [l[i] for l, i in zip(lengths, correct_idxs)]
	lengths_all = [l for lst in lengths for l in lst]
	print(f"Median length of all choices {np.median(lengths_longest):.0f}")
	print(f"Median length of correct choices {np.median(lengths_all):.0f}")







f_save = dir_results / "df_choices_with_llm_preds.csv"
df.to_csv(f_save)
ipdb.set_trace()
pass
