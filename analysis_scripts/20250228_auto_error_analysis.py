"""
python -m ipdb analysis_scripts/20250228_auto_error_analysis.py
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel

from models.openai_api import call_gpt_batch

# define the response schema
class TagResponse(BaseModel):
    tag_name: str
    justification: str

### prompts for error tagging ###
prompt_error_tag = """
Below is a multiple choice question with options and the reasoning that lead a model to an incorrect response. Your task is to use the reasoning trace to tag the error type:
1. Perception: the image was not interpreted correctly.
2. Misconception: a key concept was misunderstood.
3. Overgeneralization: the details of the question were ignored and the general case was applied.
4. Hallucination: details were added during reasoning that weren't in the question."

Question:
{{question}}

Correct answer: {{correct_answer}}

Resoning trace:
{{reasoning}}

Return a json with the following schema, the tag_name is the name of the type:
""" + json.dumps(TagResponse.schema(), indent=2)

prompt_correct_tag = """
Below is a multiple choice question with options and the reasoning that lead a model to a correct response. Your task is to use the reasoning trace and determine if the question was easy to answer because of these reasons:
1. Weak distractors: the distractors are easy to rule out.
2. Visual giveway: the question can be answered without the image because the visual information is in the question.
3. Language shortcut: the question gives away the answer in the phrasing. 
4. None: the question is not easy to answer.

Question:
{{question}}

Correct answer: {{correct_answer}}

Resoning trace:
{{reasoning}}

Return a json with the following schema, the tag_name is the name of the type:
""" + json.dumps(TagResponse.schema(), indent=2)

def replace_prompt(question, correct_idx, reasoning, is_correct=True):
    if is_correct:
        template  = prompt_correct_tag
    else:
        template = prompt_error_tag
    prompt = template.replace("{{question}}", question)
    prompt = prompt.replace("{{correct_answer}}", str(correct_idx))
    prompt = prompt.replace("{{reasoning}}", reasoning)
    return prompt

def plot_hist(df, title, save_path):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='tag_name', order=df['tag_name'].value_counts().index, palette='viridis')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

def plot_pie_chart(df, title, save_path):
    plt.figure(figsize=(8, 8))
    data = df['tag_name'].value_counts()
    colors = sns.color_palette('viridis', len(data))
    plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.85)
    plt.title(title)
    plt.gca().add_artist(plt.Circle((0, 0), 0.70, fc='white'))  # Draw a white circle at the center to make it look like a donut chart
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

data_path = '/pasteur/data/microchat/error_tagging/gpt-4o-results.csv'
save_path = '/pasteur/data/microchat/error_tagging'
model = "gpt-4o-2024-08-06" # "o1-mini-2024-09-12" "anthropic/claude-3.5-sonnet"
model_name = "gpt-4o"

os.makedirs(save_path, exist_ok=True)

# read the csv with the questions and reasoning traces
df = pd.read_csv(data_path)

# update all prompts
prompts = []
json_modes = [TagResponse for _ in range(len(df))]
for idx, row in df.iterrows():
    # remove intro from question
    question = row['question'].split('"The answer is (X)" at the end.')[1].strip()
    correct_idx = row['gt'] + 1 # 1-indexed
    reasoning = row['response']
    is_correct = row['is_correct']
    prompt = replace_prompt(question, correct_idx, reasoning, is_correct)
    prompts.append(prompt)

# call the models on all prompts
res = call_gpt_batch(prompts, model=model, json_mode=json_modes)
cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
print(f"Cost ${cost:.2f} with model {model}")
tags = [r[0]['tag_name'] for r in res]
justifications = [r[0]['justification'] for r in res]
df[f"tag_name"] = tags
df[f"justification"] = justifications
df_save_path = os.path.join(save_path, f"tagged_results_{model_name}.csv")
df.to_csv(df_save_path, index=False)

# plot a histogram of the tags
# filter by the is_correct questions
correct_df = df[df['is_correct']]
error_df = df[~df['is_correct']]
fig_save_path = os.path.join(save_path, f"correct_tags_{model_name}.png")
plot_pie_chart(correct_df, "Correct question tags", fig_save_path)
fig_save_path = os.path.join(save_path, f"error_tags_{model_name}.png")
plot_pie_chart(error_df, "Error question tags", fig_save_path)