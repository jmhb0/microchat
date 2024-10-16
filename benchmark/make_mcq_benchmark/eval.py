"""
"""
import ipdb
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import random
import json
from PIL import Image
import re

sys.path.insert(0, '.')
from models.openai_api import call_gpt_batch

dir_this_file = Path(__file__).parent

prompt_eval_templates = {
    0: {
        "about":
        "based on prompt from MMLU-pro https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/b7b9ffd84b2c21a5bfcf174fc65e5e6d74ca09a8/evaluate_from_api.py",
        "template": """\
The following is a multiple choice question (with answers). 
Think step by step and then output the answer in the format of \"The answer is (X)\" at the end." \


{{QUESTION}}

Options:
{{CHOICES}}
""",
        "regex_pattern": r"answer is \(?([0-9])\)?",
    },
    1: {
        "about":
        "based on prompt 0, but no images provided",
        "template": """\
The following is a multiple choice question (with answers).\
If an image is mentioned ignore this information and try your best to answer the question.
Think step by step and then output the answer in the format of \"The answer is (X)\" at the end."


{{QUESTION}}

Options:
{{CHOICES}}
""",
        "regex_pattern": r"answer is \(?([0-9])\)?",
    }
}


def eval_qa(key_form,
            key_question_gen,
            key_choices_gen,
            key_prompt_eval=0,
            model='gpt-4o-mini',
            seed=0,
            verbose=False):
    """ 
    Run eval - both with and without the multi-choice options. 
    """

    f_choices = Path(
        f"benchmark/data/formdata_{key_form}/question_strategy_{key_question_gen}/df_questions_key_choices_{key_choices_gen}.csv"
    )
    f_eval_closed = Path(
        f"benchmark/data/formdata_{key_form}/question_strategy_{key_question_gen}/df_questions_key_choices_{key_choices_gen}_evalclosed_{model}.csv"
    )

    if 'gpt-4o' not in model:
        raise ValueError()

    df_questions = pd.read_csv(f_choices, index_col="key_question")

    batch_prompts_text = []
    # batch_prompts_text_no_G = []
    batch_prompts_imgs = []
    idxs = []
    # question_nums = []
    gts = []
    cache_images = {}

    for idx, row in df_questions.iterrows():

        # get the images
        key_image = row['key_image']
        if key_image in cache_images.keys():
            imgs = cache_images[key_image]
        else:
            filenames = ast.literal_eval(row['fname_images'])
            try:
                imgs_pil = [Image.open(f).convert('RGB') for f in filenames]
                imgs = [np.array(img) for img in imgs_pil]
                cache_images[key_image] = imgs
                if verbose:
                    print(idx, [img.shape for img in imgs])

            except Exception as e:
                print(f"/nIssue with files {filenames}")
                print(e)
                continue
        batch_prompts_imgs.append(imgs)

        # construct the choices
        choices = ast.literal_eval(row['choices'])
        choices_str = ""
        for i, ch in enumerate(choices['choices']):
            choices_str += f"  ({i+1}): {ch}\n"

        # construct the text prompt
        prompt = prompt_eval_templates[key_prompt_eval]['template']
        prompt = prompt.replace("{{CHOICES}}", choices_str)
        prompt = prompt.replace("{{QUESTION}}", row['question'])
        batch_prompts_text.append(prompt)
        correct_index = choices['correct_index']
        gts.append(correct_index)

        # save the indexes
        idxs.append(idx)

    assert len(batch_prompts_text) == len(batch_prompts_imgs)
    assert len(batch_prompts_text) == len(idxs)

    # call gpt
    seeds = [seed] * len(batch_prompts_text)
    # blind experiment change
    if key_prompt_eval == 1:
        batch_prompts_imgs = None
    responses = call_gpt_batch(texts=batch_prompts_text,
                               imgs=batch_prompts_imgs,
                               model=model,
                               json_mode=False,
                               seeds=seeds)
    cost = sum([c[1] for c in responses])
    msgs = [m[0] for m in responses]
    print(f"Cost of vlm call w choices${cost:.3f}")

    # regex out the predictions
    # preds_letter = []
    preds = []
    for msg in msgs:
        pattern = prompt_eval_templates[key_prompt_eval]["regex_pattern"]
        match = re.search(pattern, msg)
        if match is not None:
            pred = int(match.group(1)) - 1
            preds.append(pred)
        else:
            preds.append(-1)

    # save response 
    df_questions.loc[idxs, 'gpt_response'] = msgs
    df_questions.loc[idxs, 'gpt_prediction'] = preds
    df_questions.to_csv(f_eval_closed)

    # compute the basic stats
    gts = np.array(gts)
    preds = np.array(preds)
    acc = (gts==preds).sum() / len(gts)
    print(acc)

    # todo: free LLM response


if __name__ == "__main__":
    # which form we collect the quetions from
    key_form = 0
    # which set of questions to get - made in make_questions.py
    key_question_gen = 3
    # key for generating the choices
    key_choices_gen = 3
    # key for evaluation prompt
    key_prompt_eval = 0 # 1 is blind 0 is default
    model = "gpt-4o-2024-08-06"

    eval_qa(key_form, key_question_gen, key_choices_gen,
            key_prompt_eval=key_prompt_eval, seed=0, model=model)
    ipdb.set_trace()
    pass
