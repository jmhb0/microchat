"""
python -m ipdb analysis_scripts/20250228_auto_error_analysis.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel
import datasets
from adjustText import adjust_text

from models.openai_api import call_gpt_batch

# define the response schema
class TagResponse(BaseModel):
    tag_name: str
    justification: str

### prompts for error tagging ###
prompt_error_tag = {
    0:
"""
Below is a multiple choice question with options and the reasoning that lead a model to an incorrect response. Originally the model was also shown an image with the question. Your task is to use the reasoning trace to tag the error type:
- Perception: the image was not interpreted correctly.
- Overgeneralization: the details of the question were ignored and the general case was applied.
- Hallucination: details were added during reasoning that weren't in the question or extracted from the image.
- Other: the error does not fit the above categories.

Question:
{{question}}

Correct answer: {{correct_answer}}

Resoning trace:
{{reasoning}}

Return a json with the following schema, the tag_name is the name of the type:
""" + json.dumps(TagResponse.schema(), indent=2),
1: 
# manual prompt that has the question and correct answer formatted
"""
Below is a multiple choice question with options and the reasoning that lead a model to an incorrect response. Originally the model was also shown an image with the question. Your task is to use the reasoning trace to tag the error type:
- Perception: the image was not interpreted correctly.
- Overgeneralization: the details of the question were ignored and the general case was applied.
- Hallucination: details were added during reasoning that weren't in the question or extracted from the image.
- Other: the error does not fit the above categories.

{{question}}

Resoning trace:
{{reasoning}}

Return a json with the following schema, the tag_name is the name of the type:
""" + json.dumps(TagResponse.schema(), indent=2),

2: 
# manual prompt + reasoning category
"""
Below is a multiple choice question with options and the reasoning that lead a model to an incorrect response. Originally the model was also shown an image with the question. Your task is to use the reasoning trace to tag the error type:
- perception: the image was not interpreted correctly.
- overgeneralization: the details of the question were ignored and the general case was applied.
- hallucination: details were added during reasoning that weren't in the question or extracted from the image.
- reasoning: a mistake in logical thinking that leads to an incorrect conclusion, such as misidentifying cause and effect, invalid assumptions, or flawed arguments.
- other: the error does not fit the above categories.

{{question}}

Resoning trace:
{{reasoning}}

Return a json with the following schema, the tag_name is the name of the type:
""" + json.dumps(TagResponse.schema(), indent=2),
3: # no image ablation
"""
Below is a multiple choice question with options and the reasoning that lead a model to an incorrect response. Originally the model was not shown an image with the question. Your task is to use the reasoning trace to tag the error type:
- Perception: the image was not interpreted correctly.
- Overgeneralization: the details of the question were ignored and the general case was applied.
- Hallucination: details were added during reasoning that weren't in the question or extracted from the image.
- Other: the error does not fit the above categories.

Question:
{{question}}

Correct answer: {{correct_answer}}

Resoning trace:
{{reasoning}}

Return a json with the following schema, the tag_name is the name of the type:
""" + json.dumps(TagResponse.schema(), indent=2),
}

prompt_correct_tag = {
0:
"""
Below is a multiple choice question with options and the reasoning that lead a model to a correct response. Originally the model was also shown an image with the question. Your task is to use the reasoning trace and determine if the question was answered because of these reasons:
- No image: the question can be answered without needing to interpret the image.
- Language shortcut: the question has information that makes the correct option obvious.
- Weak distractors: the distractors are easy to rule out according to the reasoning trace.
- Other: the question is hard to answer or doesn't fit the other classes.

Question:
{{question}}

Correct answer: {{correct_answer}}

Resoning trace:
{{reasoning}}

Return a json with the following schema, the tag_name is the name of the type:
""" + json.dumps(TagResponse.schema(), indent=2),
1:
"""
Below is a multiple choice question with options and the reasoning that lead a model to a correct response. Originally the model was also shown an image with the question. Your task is to use the reasoning trace and determine if the question was answered because of these reasons:
- No image: the question can be answered without needing to interpret the image.
- Language shortcut: the question has information that makes the correct option obvious.
- Weak distractors: the distractors are easy to rule out according to the reasoning trace.
- Other: the question is hard to answer or doesn't fit the other classes.

{{question}}

Resoning trace:
{{reasoning}}

Return a json with the following schema, the tag_name is the name of the type:
""" + json.dumps(TagResponse.schema(), indent=2),
2: 
"""
Below is a multiple choice question with options and the reasoning that lead a model to a correct response. Originally the model was also shown an image with the question. Your task is to use the reasoning trace and determine if the question was answered because of these reasons:
- No image: the image is unnecessary because the correct answer doesn't rely on interpreting visual cues.
- Visual giveaway: the image is unnecessary because critical visual information is already described in the question itself.
- Language shortcut: the question has information that makes the correct option obvious.
- Weak distractors: the distractors are easy to rule out according to the reasoning trace.
- Other: the question is hard to answer or doesn't fit the other classes.

Question:
{{question}}

Correct answer: {{correct_answer}}

Resoning trace:
{{reasoning}}

Return a json with the following schema, the tag_name is the name of the type:
""" + json.dumps(TagResponse.schema(), indent=2),
3: # no image ablation
"""
Below is a multiple choice question with options and the reasoning that lead a model to a correct response. Originally the model was not shown an image with the question. Your task is to use the reasoning trace and determine if the question was answered because of these reasons:
- No image: the question can be answered without needing to interpret the image.
- Language shortcut: the question has information that makes the correct option obvious.
- Weak distractors: the distractors are easy to rule out according to the reasoning trace.
- Other: the question is hard to answer or doesn't fit the other classes.

Question:
{{question}}

Correct answer: {{correct_answer}}

Resoning trace:
{{reasoning}}

Return a json with the following schema, the tag_name is the name of the type:
""" + json.dumps(TagResponse.schema(), indent=2),
}

def replace_prompt(question, correct_idx, reasoning, is_correct=True, key_prompt_error=0, key_prompt_correct=0):
    if is_correct:
        template  = prompt_correct_tag[key_prompt_correct]
    else:
        template = prompt_error_tag[key_prompt_error]

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

def plot_pie_chart(df, title, save_path, col_name='tag_name', color_name='viridis'):
    fig, ax = plt.subplots(figsize=(8, 9))  # Extra height to avoid label clutter

    # Count occurrences of each tag
    data = df[col_name].value_counts()
    total = data.sum()
    
    # Create labels with both count and percentage
    labels = [f"{tag}\n{count} ({count/total:.1%})" for tag, count in zip(data.index, data.values)]
    
    # Define colors
    colors = sns.color_palette(color_name, len(data))
    
    # Explode small slices to avoid overlap
    explode = [0.1 if count/total < 0.1 else 0 for count in data.values]

    # Plot pie chart with leader lines
    wedges, texts, autotexts = ax.pie(
        data, startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}, explode=explode,
        autopct='', pctdistance=0.85
    )

    # Adjust label positions with leader lines
    for wedge, label in zip(wedges, labels):
        angle = (wedge.theta2 + wedge.theta1) / 2  # Midpoint angle of the wedge
        x = np.cos(np.radians(angle)) * 1.4  # Move text further out
        y = np.sin(np.radians(angle)) * 1.4
        
        # Add text label
        ax.text(x, y, label, ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))

        # Draw a leader line from the wedge to the text
        ax.plot([np.cos(np.radians(angle)), x], [np.sin(np.radians(angle)), y], 
                color='gray', linestyle='dotted', linewidth=0.8)

    fig.suptitle(title, fontsize=14, fontweight='bold')  # Moves title well above chart

    # Convert to a donut chart
    ax.add_artist(plt.Circle((0, 0), 0.70, fc='white'))

    fig.subplots_adjust(top=0.85)  # Adds space above the chart so the title is visible
    plt.savefig(save_path, dpi=200)

def plot_pie_chart_all(df, title, save_path, col_name='tag_name', by_name='is_correct'):
    fig, ax = plt.subplots(figsize=(8, 9))  # Extra height to avoid label clutter

    # replace other with other_correct and other_wrong
    df.loc[(~df[by_name]) & (df[col_name] == 'Other'), col_name] = 'Other_wrong'
    df.loc[(df[by_name]) & (df[col_name] == 'Other'), col_name] = 'Other_correct'

    # Count occurrences of each tag separately for sorting
    correct_data = df[df[by_name] == True][col_name].value_counts()
    wrong_data = df[df[by_name] == False][col_name].value_counts()
    

    # Concatenate them while maintaining order
    data = pd.concat([correct_data, wrong_data])

    total = data.sum()

    # Create labels with both count and percentage
    labels = [f"{tag}\n{count} ({count/total:.1%})" for tag, count in zip(data.index, data.values)]

    # Get the number of unique categories in each group
    num_correct = correct_data.nunique()
    num_wrong = wrong_data.nunique()

    # Define color palettes based on the unique number of categories
    palette_correct = sns.color_palette('Blues', num_correct)
    palette_wrong = sns.color_palette('Reds', num_wrong)

    # Create a mapping from category to color
    correct_tags = correct_data.index
    wrong_tags = wrong_data.index

    colors_map = {tag: palette_correct[i] for i, tag in enumerate(correct_tags)}
    colors_map.update({tag: palette_wrong[i] for i, tag in enumerate(wrong_tags)})

    colors = [colors_map[tag] for tag in data.index]

    # Explode small slices to avoid overlap
    explode = [0.1 if count/total < 0.1 else 0 for count in data.values]

    # Plot pie chart with leader lines
    wedges, texts, autotexts = ax.pie(
        data, startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}, explode=explode,
        autopct='', pctdistance=0.85
    )

    # Adjust label positions with leader lines
    for wedge, label in zip(wedges, labels):
        angle = (wedge.theta2 + wedge.theta1) / 2  # Midpoint angle of the wedge
        x = np.cos(np.radians(angle)) * 1.4  # Move text further out
        y = np.sin(np.radians(angle)) * 1.4

        # Add text label
        ax.text(x, y, label, ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))

        # Draw a leader line from the wedge to the text
        ax.plot([np.cos(np.radians(angle)), x], [np.sin(np.radians(angle)), y], 
                color='gray', linestyle='dotted', linewidth=0.8)

    fig.suptitle(title, fontsize=14, fontweight='bold')  # Moves title well above chart

    # Convert to a donut chart
    ax.add_artist(plt.Circle((0, 0), 0.70, fc='white'))

    fig.subplots_adjust(top=0.85)  # Adds space above the chart so the title is visible
    plt.savefig(save_path, dpi=200)

def run_tagging(df, save_path, model, df_save_path, remove_intro=False, key_prompt_error=0, key_prompt_correct=0):
    # log the correct and incorrect prompt templates
    prompt_save_path = os.path.join(save_path, "prompt_templates.txt")
    with open(prompt_save_path, 'w') as f:
        f.write(prompt_correct_tag[key_prompt_correct])
        f.write(prompt_error_tag[key_prompt_error])

    # update all prompts
    prompts = []
    json_modes = [TagResponse for _ in range(len(df))]
    for idx, row in df.iterrows():
        question = row['question']
        # remove intro from question
        if remove_intro:
            question = question.split('"The answer is (X)" at the end.')[1].strip()
        correct_idx = row['gt'] + 1 # 1-indexed
        reasoning = row['response']
        is_correct = row['is_correct']
        prompt = replace_prompt(question, correct_idx, reasoning, is_correct, key_prompt_error, key_prompt_correct)
        prompts.append(prompt)

    # call the models on all prompts
    res = call_gpt_batch(prompts, model=model, json_mode=json_modes)
    cost = sum([r[1] for r in res])  # with GPT-4o this cost $1.4
    print(f"Cost ${cost:.2f} with model {model}")
    tags = [r[0]['tag_name'] for r in res]
    justifications = [r[0]['justification'] for r in res]
    df[f"tag_name"] = tags
    df[f"justification"] = justifications
    df.to_csv(df_save_path, index=False)
    return df

data_path = '/pasteur/data/microchat/error_tagging/gpt-4o-results.csv'
save_path = '/pasteur/data/microchat/error_tagging/v1.4'
# data_path = '/pasteur/data/microchat/error_tagging/claude-no-image-ablation-results.csv'
# save_path = '/pasteur/data/microchat/error_tagging/v1.4_no_image_ablation'
model = "gpt-4o-2024-08-06" # "o1-mini-2024-09-12" "anthropic/claude-3.5-sonnet"
model_name = "gpt-4o"
key_prompt_error = 3 #0 
key_prompt_correct = 3 #2

os.makedirs(save_path, exist_ok=True)

# read the csv with the questions and reasoning traces
df = pd.read_csv(data_path)
# join information from the og dataset.
# TODO: add the columns to the og saved csv
dataset = datasets.load_dataset("jmhb/microvqa")['train']
df['key_question'] = dataset['key_question']

df_save_path = os.path.join(save_path, f"tagged_results_{model_name}.csv")
if os.path.exists(df_save_path):
    print(f"Loading the tagged results from {df_save_path}")
    df = pd.read_csv(df_save_path)
else:
    df = run_tagging(df, save_path, model, df_save_path, remove_intro=True,
                     key_prompt_error=key_prompt_error, key_prompt_correct=key_prompt_correct)
# plot a histogram of the tags
correct_df = df[df['is_correct']]
error_df = df[~df['is_correct']]

fig_save_path = os.path.join(save_path, f"correct_tags_{model_name}.png")
plot_pie_chart(correct_df, "Correct question tags", fig_save_path)
fig_save_path = os.path.join(save_path, f"error_tags_{model_name}.png")
plot_pie_chart(error_df, "Error question tags", fig_save_path)
fig_save_path = os.path.join(save_path, f"all_tags_{model_name}.png")
plot_pie_chart_all(df, "All question tags", fig_save_path)
import ipdb; ipdb.set_trace()
correct_df['tag_name'].value_counts()
error_df['tag_name'].value_counts()

# # evaluate the correct tags with Jeff's manual annotated ones
# manual_tag_path = '/pasteur/data/microchat/error_tagging/manual_tagging_eval_anthropicclaude-35-sonnet_naive.csv'
# key_prompt_error = 1
# key_prompt_correct = 1

# manual_save_path = save_path + '_manual'
# manual_df_save_path = os.path.join(manual_save_path, f"tagged_results_{model_name}_manual.csv")
# os.makedirs(manual_save_path, exist_ok=True)

# manual_df = pd.read_csv(manual_tag_path)
# # keep only the columns we need
# manual_df['is_correct'] = (manual_df['pred'] == manual_df['gt'])

# # rename some of the columns
# manual_df = manual_df[['key_question', 'error_category', 'error_comment', 'error_rationale', 'question_answer_2_formatted', 'msg', 'is_correct', 'gt', 'pred']]
# manual_df.rename(columns={'question_answer_2_formatted': 'question', 'msg': 'response'}, inplace=True)
# eval_df = run_tagging(manual_df, manual_save_path, model, manual_df_save_path, remove_intro=False,
#                       key_prompt_error=key_prompt_error, key_prompt_correct=key_prompt_correct)

# # simplify some error categories
# eval_df['error_category'].replace(['misconception', 'reasoning'], ['overgeneralization', 'other'], inplace=True)

# # plot a histogram of the tags
# correct_df = eval_df[eval_df['is_correct']]
# error_df = eval_df[~eval_df['is_correct']]
# fig_save_path = os.path.join(manual_save_path, f"correct_tags_{model_name}_all.png")
# plot_pie_chart(correct_df, "Correct question tags", fig_save_path, color_name='crest')
# fig_save_path = os.path.join(manual_save_path, f"error_tags_{model_name}_all.png")
# plot_pie_chart(error_df, "Error question tags", fig_save_path, color_name='flare')
# fig_save_path = os.path.join(manual_save_path, f"all_tags_{model_name}_all.png")
# plot_pie_chart_all(eval_df, "All question tags", fig_save_path)

# # filter to those that have been tagged
# eval_df = eval_df[eval_df['error_category'].notnull()]

# # compare tag_name and error_category lowercase
# eval_df['tag_name'] = eval_df['tag_name'].str.lower()
# # plot manual eval
# correct_df = eval_df[eval_df['is_correct']]
# error_df = eval_df[~eval_df['is_correct']]

# fig_save_path = os.path.join(manual_save_path, f"error_tags_manual_manualset.png")
# plot_pie_chart(error_df, "Error question tags", fig_save_path, col_name='error_category')
# fig_save_path = os.path.join(manual_save_path, f"error_tags_{model_name}_manualset.png")
# plot_pie_chart(error_df, "Error question tags", fig_save_path, col_name='tag_name')
# fig_save_path = os.path.join(manual_save_path, f"all_tags_{model_name}_manualset.png")
# plot_pie_chart_all(eval_df, "All question tags", fig_save_path)

# # calculate the accuracy
# correct = eval_df['tag_name'] == eval_df['error_category']
# eval_df['tag_correct'] = correct
# accuracy = (correct.sum() / len(eval_df)) * 100
# print(f"Accuracy of the model: {accuracy:.2f}")
# # save eval_df to file
# eval_df.to_csv(manual_df_save_path, index=False)
# # # use existing tagged df
# # # eval_df = pd.merge(df[~df['is_correct']], manual_df, on='key_question', how='inner')