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
    }
}


def eval_qa(key_form,
            key_question_gen,
            key_choices_gen,
            model='gpt-4o-mini',
            seed=0,
            verbose=False):
    """ 
    Run eval - both with and without the multi-choice options. 
    """
    key_prompt_eval = 0

    f_choices = Path(
        f"benchmark/data/formdata_{key_form}/question_strategy_{key_question_gen}/df_questions_key_choices_{key_choices_gen}.csv"
    )

    if 'gpt-4o' not in model:
        raise ValueError()

    df_questions = pd.read_csv(f_choices)

    batch_prompts_text = []
    # batch_prompts_text_no_choices = []
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

    gts = np.array(gts)
    preds = np.array(preds)
    acc = (gts==preds).sum() / len(gts)
    print(acc)
    ipdb.set_trace()

    pass

    # # make `vqas_results` to save the llm responses, and also delete the images
    # vqas_results = copy.deepcopy(vqas)
    # for i in range(len(idxs)):
    #     idx = idxs[i]
    #     question_num = question_nums[i]

    #     del vqas_results[idx][question_num]['imgs']
    #     vqas_results[idx][question_num]['response'] = msgs[i]
    #     vqas_results[idx][question_num]['response_no_choices'] = msgs_no_choices[i]
    #     vqas_results[idx][question_num]['pred'] = preds[i]

    
    # return vqas_results, gts, preds, acc


if __name__ == "__main__":
    # which form we collect the quetions from
    key_form = 0
    # which set of questions to get - made in make_questions.py
    key_question_gen = 0
    # key for generating the choices
    key_choices_gen = 1
    model = 'gpt-4o-mini'

    eval_qa(key_form, key_question_gen, key_choices_gen, seed=0, model=model)
    ipdb.set_trace()
    pass
