"""
python analysis_scripts/20241112_llm_tagging_final.py

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
import os
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
import tiktoken

sys.path.insert(0, "..")
sys.path.insert(0, ".")

from models.openai_api import call_gpt_batch

# https://platform.openai.com/docs/models/gpt-4o
model = "gpt-4o-2024-08-06"


version_name = '2'
date_version = 'nov11'
version_dir = f'/pasteur/data/microchat/dataset_versions/{date_version}'
dir_results = os.path.join(version_dir, f'tagging_results_{version_name}')
os.makedirs(dir_results, exist_ok=True)
bio_tag_path = os.path.join(version_dir, f'bio_tags_{date_version}_stage2.csv')
blooms_tag_path = os.path.join(version_dir, 'blooms_tagging', f'ours_{date_version}_stage2', 'blooms_tags_0_0.csv')

## data
# get the dataset with 
f_questions_choices = os.path.join(version_dir, f'benchmark_{date_version}_stage2.csv')
df = pd.read_csv(f_questions_choices)
if 'key_question.1' in df.columns:
    df.drop(columns=['key_question.1'], inplace=True)
# also the images and questions dataframes
df_images = pd.read_csv(os.path.join(version_dir,  f"4_images_benchmark_{date_version}.csv"))
df_questions = pd.read_csv(os.path.join(version_dir, f"4_questions_benchmark_{date_version}.csv"))
# select relevant columns
if 'key_image.1' in df_images.columns:
    df_images.drop(columns=['key_image.1'], inplace=True)
df_questions = df_questions[['key_question', 'key_image', 'use_case', 'comments', 'incorrect_answer']]
# merge into a single dataframe
df = df.merge(df_images, left_on='key_image', right_on='key_image', how='left')
df = df.merge(df_questions, left_on=['key_question', 'key_image'], right_on=['key_question', 'key_image'], how='left')

# create column for joint question and answer from final version
# find correct answer from choices
df[f'answer_{version_name}'] = [ast.literal_eval(row[f'choices_{version_name}'])[row[f'correct_index_{version_name}']] for _, row in df.iterrows()]

df[f'question_and_answer_{version_name}'] = [
    f"Question:\n```{q}```\n\nAnswer:\n```{a}```"
    for q, a in zip(df[f'question_{version_name}'], df[f'answer_{version_name}'])]

df[f'description_question_answer_{version_name}'] = [
    f"Description of image preparation:\n```{d}```\n\nQuestion:\n```{q}```\n\nAnswer:\n```{a}```"
    for d, q, a in zip(df[f'Context - image generation'], df[f'question_{version_name}'], df[f'answer_{version_name}'])]


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
if 0:
    print("Use case generation")
    json_mode = True
    prompts = [
        prompt_template_usecase.replace("{{text}}", t)
        for t in df[f'description_question_answer_{version_name}']
    ]
    res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
    cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
    print(f"Cost ${cost:.2f} with model {model}")
    use_cases = [r[0]['use_case'] for r in res]
    df[f'_use_case_{version_name}'] = use_cases

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
        for t in df['Context - image generation']
    ]
    res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
    cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
    print(f"Cost ${cost:.2f} with model {model}")
    df['modality'] = [r[0]['modality'] for r in res]
    df['scale'] = [r[0]['scale'] for r in res]
    # assert len(res) == len(df_images)
    # df['_image_modality'] = df['key_image'].map(
    #     df_images['modality'].to_dict())
    # df['_image_scale'] = df['key_image'].map(df_images['scale'].to_dict())

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
	"We see something unusual: is it biologically meaningful, or an experimental artifact?"

Respond in json: {"sub_use_case" : "1.1|1.2"}

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
	"why does increased laser power lead to more release of mScarlet-CD4 from the golgi to the cell membrane?"

You can pick one option only.
Respond in json: {"sub_use_case" : "2.1|2.2"}

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
    for use_case in (1, 2, 3):
        idxs = df[df['use_case'] == use_case].index
        prompts = [
            prompt_templates_usecase_lvl2[str(use_case)].replace("{{text}}", t)
            for t in df.loc[idxs, f'question_and_answer_{version_name}']
        ]
        res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
        cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
        print(
            f"Cost for use-case {use_case} with ${cost:.2f} with model {model}"
        )
        sub_use_cases = [r[0]['sub_use_case'] for r in res]
        df.loc[idxs, f'_sub_use_case{version_name}'] = sub_use_cases

# 2. Experimental misunderstanding: misapplying relevant concepts or misinterpreting the impact of experimental parameters.
# 3. Misleading reasoning: is an error due to flawed reasoning, (e.g. incorrect cause-effect assumptions, reversals).
prompt_template_distractor_types = """
Below is a text that is paired with a microscopy image. 
The text is a question and answer choices where only one is correct.

Your task is to identify what the distractors are testing compared to the correct answer from these categories:
1. Perception error: is based on an incorrect interpretation of the image.
2. Conceptual misunderstanding: is a gap in fundamental knowledge or theory.
3. Oversimplification or Overgeneralization: Applying a general principle too broadly without accounting for context-specific nuances.
4. Cause-Effect Misinterpretation: Misunderstanding or incorrectly assuming a cause-and-effect relationship.
5. Irrelevant technical details: technically plausible-sounding but unrelated to the question's actual scientific context.

The output json should be:
{'distractor_type' : ['1|2|3|4|5', ...], 'explanation' : [explanation distractor 1, ...]}

Here is the question, answer, and distractors:
```
Question:
{{text_question}}
Correct Answer:
{{text_correct_answer}}
Distractors:
{{text_choices}}
```"""
if 1:
    json_mode = True
    print("\nRunning 'tagging distractors' test")
    prompts = []
    for t_q, t_c, t_a in zip(df[f'question_{version_name}'], df[f'choices_{version_name}'], df[f'answer_{version_name}']):
        t_c = ast.literal_eval(t_c)
        # remove correct answer from distractors
        t_c = [c for c in t_c if c != t_a]
        prompt = prompt_template_distractor_types.replace("{{text_question}}", str(t_q))
        prompt = prompt.replace("{{text_correct_answer}}", str(t_a))
        prompt = prompt.replace("{{text_choices}}", str(t_c))
        prompts.append(prompt)
    res = call_gpt_batch(prompts, model=model, json_mode=json_mode)
    cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
    print(f"Cost ${cost:.2f} with model {model}")
    df[f'_distractor_tags_{version_name}'] = [r[0]['distractor_type'] for r in res]
    df[f'_distractor_tags_explanations_{version_name}'] = [
        r[0]['explanation'] for r in res
    ]

# add length of distractor and correct choices to tags
if 0:
    num_incorrect = 5
    gt_idx = df[f'choices_{version_name}'].apply(lambda x: ast.literal_eval(x)[f'correct_index_{version_name}']).to_numpy()
    choices = np.vstack(df[f'choices_{version_name}'].apply(lambda x: ast.literal_eval(x)[f'choices_{version_name}']))
    df[f'correct_length_{version_name}'] = np.char.str_len(choices[np.arange(len(choices)), gt_idx])
    mask = np.ones_like(choices, dtype=bool)
    mask[np.arange(len(choices)), gt_idx] = False
    df[f'mean_distractor_length_{version_name}'] = np.mean(np.char.str_len(choices[mask].reshape(-1, num_incorrect)), axis=1)
    df[f'correct_is_longer_{version_name}'] = ((df[f'correct_length_{version_name}'] > df[f'mean_distractor_length_{version_name}']) * 1).astype(str)
    # token length tags
    enc = tiktoken.encoding_for_model("gpt-4o")
    vec_encode_len = np.vectorize(lambda x: len(enc.encode(x)))
    df[f'correct_token_length_{version_name}'] = vec_encode_len(choices[np.arange(len(choices)), gt_idx])
    df[f'mean_distractor_token_length_{version_name}'] = np.mean(vec_encode_len(choices[mask].reshape(-1, num_incorrect)), axis=1)

# summarize all tag metrics
if 0:
    def count_tag_instances(df, tag_name):
        num = df[tag_name].value_counts().loc['1']
        print(f'| {tag_name} | {num} | {num/len(df):.2f} |')

    # count_tag_instances(df, '_question_has_answer')
    # count_tag_instances(df, '_question_no_image')
    # count_tag_instances(df, '_question_convoluted')
    # count_tag_instances(df, '_question_basic')
    # count_tag_instances(df, '_question_bad_grammar')
    # count_tag_instances(df, '_answer_generation_is_different')
    count_tag_instances(df, f'correct_is_longer_{version_name}')

bio_ext = ''
if os.path.exists(bio_tag_path):
    print(f'loading bio tags from {bio_tag_path}')
    df_bio = pd.read_csv(bio_tag_path)
    # keep only relevant columns
    df_bio = df_bio[['key_image', 'key_question', 'organism', 'specimen', 'research_subject', 'research_subject_list',
       'organism_research_rationale', 'consensus_difficulty', 'fk_difficulty', 'fk_reading_ease']]
    df = df.merge(df_bio, left_on=['key_image', 'key_question'], right_on=['key_image', 'key_question'], how='left')
    bio_ext = '_bio'

blooms_ext = ''
if os.path.exists(blooms_tag_path):
    print(f'loading blooms tags from {blooms_tag_path}')
    df_blooms = pd.read_csv(blooms_tag_path)
    # keep only relevant columns
    df_blooms = df_blooms[['key_question', 'blooms_level', 'blooms_reasoning', 'blooms_name']]
    df = df.merge(df_blooms, left_on=['key_question'], right_on=['key_question'], how='left')
    blooms_ext = '_blooms'

f_save = os.path.join(dir_results, f"tagged_{date_version}_{version_name}{bio_ext}{blooms_ext}.csv")
print(f'saving to {f_save}')
df.to_csv(f_save)

