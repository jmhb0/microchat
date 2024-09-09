"""
Use LLMs for certain annotations. These annotation changes should all be human-verified. 

They are:
- for multiple images, the context string refers to filenames. Change it to refer to idx_0, idx_1 instead. 
- whether a question is a follow up to another question
- 'use case' classification. Try to catch errors. 
"""
import sys
import ipdb
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from models.openai_api import call_gpt_batch

dir_this_file = Path(__file__).parent


def get_dfs(idx_form):
    dir_data = Path(f"benchmark/data/formdata_{idx_form}")
    dir_data.mkdir(exist_ok=True, parents=True)
    df_people = pd.read_csv(dir_data / '2_df_people.csv', index_col='key_person')
    df_images = pd.read_csv(dir_data / '2_df_images.csv', index_col='key_image')
    df_questions = pd.read_csv(dir_data / '2_df_questions.csv', index_col='key_question')

    return df_images, df_people, df_questions


def llm_image_checks(idx_form, df_images):
    # llm call to get the new fields
    df_images = llm_image_context_update(idx_form, df_images)

    # columns that will be useful for annotation
    df_images['image_edits'] = ""  # notes for image edits that are needed
    df_images['new_image_url'] = ""
    df_images['num_panels'] = "1,"

    # shuffle, also for annotation reasons
    cols = df_images.columns.tolist()

    # Remove the two columns we want at the end
    cols.remove('Context - image generation')
    cols.remove('contexts_updated')
    cols = cols + ['Context - image generation', 'contexts_updated']
    df_images = df_images[cols]

    dir_data = Path(f"benchmark/data/formdata_{idx_form}")
    # put in dataframe and save to csv
    f_save = dir_data / "3_images_updates.csv"
    df_images.to_csv(f_save)


def llm_image_context_update(idx_form, df_images):
    # multiimage images
    mask_multiimage = df_images['image_counts'] > 1
    idxs_multiimage = np.where(mask_multiimage)[0]

    # context information
    contexts = df_images.loc[idxs_multiimage, 'Context - image generation']
    filenames = df_images.loc[idxs_multiimage, 'fnames_images']

    # create a text prompt
    prompt_template = """
Below is a list of filenames in alphabetical order and a text string.
The text might contain references to the filenames, but their may be small errors in the spelling of the filename.
If the 'text' refers to the filenames, then replace each instance of the filename with {img_0}, {img_1}, .... where the numbers are the index of that filename.
If the text does not refer to the filenames then just return '0'. 
Return a json like {"updated_text" : "..."}

Filenames: {{filenames}}. 
Text: 
```
{{text}}
```"""
    prompts_batch = []
    for (fs, t) in zip(filenames, contexts):
        prompt = prompt_template.replace("{{filenames}}",
                                         fs).replace("{{text}}", t)
        prompts_batch.append(prompt)

    # run
    res = call_gpt_batch(prompts_batch, json_mode=True)
    contexts_updated = [r[0]['updated_text'] for r in res]

    # add to the relevant
    df_images['contexts_updated'] = '0'
    df_images.loc[idxs_multiimage, 'contexts_updated'] = contexts_updated

    return df_images


def llm_questions_check(idx_form, df_questions, df_images):
    df_questions = llm_is_follow_up_question(idx_form, df_questions)
    df_questions = llm_is_wrong_usecase(idx_form, df_questions)
    # df_questions = llm_question_image_context_references(
    #     idx_form, df_questions, df_images)
    df_questions = llm_field_image_context_references(idx_form, df_questions,
                                                      df_images, "question")
    df_questions = llm_field_image_context_references(idx_form, df_questions,
                                                      df_images, "answer")
    df_questions = llm_field_image_context_references(idx_form, df_questions,
                                                      df_images,
                                                      "incorrect_answer")
    df_questions = llm_could_be_follow_up_but_not_marked(
        idx_form, df_questions)
    dir_data = Path(f"benchmark/data/formdata_{idx_form}")
    f_save = dir_data / "3_question_updates.csv"
    df_questions.to_csv(f_save)


def llm_is_follow_up_question(idx_form, df_questions):
    df_questions['maybe_followup'] = df_questions['use_case'].isin(
        (2, 3)) & ~df_questions['comments'].isna()
    mask = df_questions['maybe_followup'] > 0
    idxs = np.where(mask)[0]
    comments = df_questions.loc[idxs, 'comments'].values

    # identify questions with comments
    prompt_template = """
Below is a piece of text. 
If it contains text with the same meaning as "follow up to question 2", return {'ans':'2'}.
If it contains text with the same meaning as "follow up to question 3", return {'ans':'3'}.
and so on for any number. 
Else return {'ans':'0'}.
The response must be json.

Text: 
```
{{text}}
```"""
    # create a text prompt
    prompts_batch = []
    for t in comments:
        prompt = prompt_template.replace("{{text}}", t)
        prompts_batch.append(prompt)

    # run
    res = call_gpt_batch(prompts_batch, json_mode=True)
    followup = [r[0]['ans'] for r in res]

    # add to the relevant
    df_questions['follow_up'] = '0'
    df_questions.loc[idxs, 'follow_up'] = followup

    return df_questions


def llm_could_be_follow_up_but_not_marked(idx_form, df_questions):
    """ 
    Some questions could be followups to a use case 1 question, but they forgot
    to mark them. Use an LLM to estimate them.

    This is run after `llm_is_follow_up_question` which populates the column 
    'follow_up' if the user said it was a follow up in their 'comments'.
    """

    df_questions['candidate_follow_up'] = False # will fill with True if True

    # first the "use case 2"
    # df_questions['maybe_nontagged_followup']
    mask_usecase2 = (df_questions['use_case']
                     == 2) | (df_questions['use_case_llm_predicted']
                              == 2) & (df_questions['follow_up'] == "0").values
    mask_usecase3 = (df_questions['use_case']
                     == 3) | (df_questions['use_case_llm_predicted']
                              == 3) & (df_questions['follow_up'] == "0")

    idxs_usecase2 = mask_usecase2[mask_usecase2>0].index.values
    idxs_usecase3 = mask_usecase3[mask_usecase3>0].index.values

    # identify questions with comments
    prompt_template = """
Below is a 'context' that refers to a microscopy image, then a 'question' about the image.
Some of these quesitons are 'follow-ups' from other questions. 
This means that the earlier question and answer has important information that makes this question make sense. 
The earlier question and answer might discuss important visual observation about an an image. 
Questions that are follow-ups will seem like they're missing some important context.

If it is a followup, return {'ans':'True'}
Else return {'ans':'False'}.
The response must be json.

CONTEXT: 
```
{{caption}}
```
QUESTION: 
```
{{question}}
```
"""
    # create a text prompt
    prompts_batch = []
    for idx in idxs_usecase2:
        question = df_questions.loc[idx, 'question']
        key_image = df_questions.loc[idx,'key_image']
        caption = df_images.loc[key_image, 'caption']
        prompt = prompt_template.replace("{{caption}}", str(caption))
        prompt = prompt.replace("{{question}}", question)
        prompts_batch.append(prompt)
    res = call_gpt_batch(prompts_batch, json_mode=True)
    followup = [True if r[0]['ans']=='True' else False for r in res]
    df_questions.loc[idxs_usecase2, 'candidate_follow_up'] = followup


    # identify questions with comments
    prompt_template = """
Below is a 'context' that refers to a microscopy image, then a 'question' about the image.
Some of these quesitons are 'follow-ups' from other questions. 
This means that the earlier question and answer has important information that makes this question make sense. 
The earlier question and answer might discuss visual features of the image, or discuss what could be causing what we see in the image.
Questions that are follow-ups will seem like they're missing some important context.

If it is a followup, return {'ans':'True'}
Else return {'ans':'False'}.
The response must be json.

CONTEXT: 
```
{{caption}}
```
QUESTION: 
```
{{question}}
```
"""
    # create a text prompt
    prompts_batch = []
    for idx in idxs_usecase3:
        question = df_questions.loc[idx, 'question']
        key_image = df_questions.loc[idx, 'key_image']
        caption = df_images.loc[key_image, 'caption']
        prompt = prompt_template.replace("{{caption}}", str(caption))
        prompt = prompt.replace("{{question}}", question)
        prompts_batch.append(prompt)
    res = call_gpt_batch(prompts_batch, json_mode=True)
    followup = [True if r[0]['ans']=='True' else False for r in res]

    df_questions.loc[idxs_usecase3, 'candidate_follow_up'] = np.logical_or(followup, df_questions.loc[idxs_usecase3, 'candidate_follow_up'])

    return df_questions


def llm_is_wrong_usecase(idx_form, df_questions):
    dir_data = dir_this_file / f"formdata_{idx_form}"

    # identify questions with comments
    prompt_template = """
Below is a text that is paired with a microscopy image. The text is a biological question.

Choose which 'use case' best describes this question. The options are:
1. What is unusual or interesting in these images? And why is it unusual or interesting? It often asks about image features. 
2. What are the possible mechanisms that could cause it? It focuses on underlying causes for image content.
3. What should we do next and why? It often suggests next steps. 
You can pick one option only 

Response in json, one option only, so one of: 
{'use_case' : "1"} or {'use_case' : "2"} or {'use_case' : "3"}

Text: 
```
{{text}}
```"""
    # create a text prompt
    prompts_batch = []
    for t in df_questions['question']:
        prompt = prompt_template.replace("{{text}}", t)
        prompts_batch.append(prompt)

    # run
    res = call_gpt_batch(prompts_batch, json_mode=True)
    use_cases = np.array([int(r[0]['use_case']) for r in res])

    df_questions['use_case_llm_predicted'] = use_cases
    df_questions['possible_updated_use_case'] = ~(use_cases
                                                  == df_questions['use_case'])

    return df_questions




def llm_field_image_context_references(idx_form, df_questions, df_images,
                                       field_mode):
    # Determine input and output fields based on mode
    if field_mode in ["question", "answer", "incorrect_answer"]:
        input_field = field_mode
        output_field = f"{input_field}_updated"
    else:
        raise ValueError(
            "Invalid field_mode. Choose 'question', 'answer', or 'incorrect_answer'."
        )

    # identify entries for images with multiple images
    image_idxs_multiimage = df_images[df_images['image_counts'] > 1].index
    mask = df_questions['key_image'].isin(image_idxs_multiimage)
    idxs = df_questions[mask].index.values

    # create a text prompt
    prompt_template = """
Below is a list of filenames in alphabetical order and a text string.
The text might contain references to the filenames, but there may be small errors in the spelling of the filename.
If the 'text' refers to the filenames, then replace each instance of the filename with {img_0}, {img_1}, .... where the numbers are the index of that filename.
If the text does not refer to the filenames then just return '0'. 
Return a json like {"updated_text" : "..."}
Filenames: {{filenames}}. 
Text: 
```
{{text}}
```"""

    prompts_batch = []
    image_counts = []
    filenames_all = []
    for idx in idxs:
        text = df_questions.loc[idx][input_field]
        # recover the filenames from the images
        idx_image = int(df_questions.loc[idx, 'key_image'])
        image_count = df_images.loc[idx_image, 'image_counts']
        image_counts.append(image_count)
        assert image_count > 1
        filenames = df_images.loc[idx_image, 'fnames_images']
        filenames_all.append(filenames)
        # make the prompt
        prompt = prompt_template.replace("{{filenames}}", filenames).replace(
            "{{text}}", str(text))
        prompts_batch.append(prompt)

    # run
    res = call_gpt_batch(prompts_batch, json_mode=True)
    updated_texts = [r[0]['updated_text'] for r in res]

    # only run this one the first time you see it
    if 'image_counts' not in df_questions.columns:
        df_questions['image_counts'] = 0
        df_questions.loc[idxs, 'image_counts'] = image_counts
        df_questions['filenames'] = ''
        df_questions.loc[idxs, 'filenames'] = filenames_all

    df_questions[output_field] = '0'
    df_questions.loc[idxs, output_field] = updated_texts

    return df_questions


if __name__ == '__main__':
    idx_form = 0
    df_images, df_people, df_questions = get_dfs(idx_form)
    llm_image_checks(idx_form, df_images)
    llm_questions_check(idx_form, df_questions, df_images)
