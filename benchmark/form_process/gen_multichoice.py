import os
import pandas as pd
import ipdb
import re
import random
import json
from PIL import Image
import numpy as np
from pathlib import Path
import copy

import sys
import logging

sys.path.insert(0, ".")

from models.openai_api import call_gpt_batch, call_gpt

logging.basicConfig(level=logging.INFO)
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# yapf: disable
prompt_templates = {
    0: {
        "about" : "basic, including context",
        "template" :"""
I’m creating a dataset to evaluate VLM understanding on biomedical images.
Could you convert this user input question into a multi-choice question with 6 answer choices? 
One choice should be "None of the above", and this choice should have a 1/6 chance of being correct.

Output a JSON format:
{{"question": str, "choices": list, "answer": int (start from 0)}}.

Context: {CONTEXT}
Input Question: {QUESTION}
Correct Answer: {ANSWER}
"""
    },
    # use the incorrect answer
    1: {
        "about" : "similar to 1 but using the incorrect answer",
        "template" :"""
I’m creating a dataset to evaluate VLM understanding on biomedical images.
Could you convert this user input question into a multi-choice question with 6 answer choices? 
We include one 'Example incorrect answer' that can be includes. 
The 'Correct answer' and 'Example incorrect answer' can be abbreviated if they're too long, and all choices should have a similar linguistic style.
One choice should be "None of the above".

Output a JSON format:
{{"question": str, "choices": list, "answer": int (start from 0)}}.

Context: {CONTEXT}
Input Question: {QUESTION}
Correct Answer: {ANSWER}
Example Incorrect Answer: {INCORRECT_ANSWER}
"""
    }
}
# yapf: enable
prompt_eval_templates = {
    0: {
        "about":
        "based on prompt from MMLU-pro https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/b7b9ffd84b2c21a5bfcf174fc65e5e6d74ca09a8/evaluate_from_api.py",
        "template":
        """\
The following is a multiple choice questions (with answers) about microscopy images. 
Think step by step and then output the answer in the format of \"The answer is (X)\" at the end." \

{QUESTION}
""",
        "regex_pattern" : r"answer is \(?([0-9])\)?",
    }
}

# yapf: disable
prompt_eval_template = {
    0: {
        "about" : "basic, including context",
        "template" :"""
I’m creating a dataset to evaluate VLM understanding on biomedical images.
Could you convert this user input question into a multi-choice question with 6 answer choices? 
One choice should be "None of the above", and this choice should have a 1/6 chance of being correct.

Output a JSON format:
{{"question": str, "choices": list, "answer": int (start from 0)}}.

Context: {CONTEXT}
Input Question: {QUESTION}
Correct Answer: {ANSWER}
"""
    }
}
# yapf: enable


def find_column(df, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    matches = [col for col in df.columns if regex.search(col)]
    return matches[0] if matches else None


def get_form_questions(key_form, dir_path):

    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} does not exist.")

    file_path = os.path.join(dir_path, "responses.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pd.read_csv(file_path)
    responses = {}

    for index, row in df.iterrows():
        row_dict = {}

        context_column = find_column(df, r'^Context')
        if context_column:
            row_dict['context'] = row[context_column]
        else:
            row_dict['context'] = None

        row_dict['questions'] = []
        row_dict['answers'] = []
        row_dict['answers_incorrect'] = []
        row_dict['use_cases'] = []

        for i in range(1, 14):
            question_col = find_column(df, rf'^Question\s*{i}\b')
            answer_col = find_column(df, rf'^Answer\s*{i}\b')
            incorrect_answer_col = find_column(
                df, rf'^Incorrect\s*Answer\s*{i}\b')
            use_case_col = find_column(
                df, rf'^Question\s*{i}\s*use\s*case(?:s)?\b')

            if question_col and pd.notna(
                    row[question_col]) and row[question_col].strip():
                row_dict['questions'].append(row[question_col])
                row_dict['answers'].append(
                    row[answer_col] if answer_col else '')
                row_dict['answers_incorrect'].append(
                    row[incorrect_answer_col] if incorrect_answer_col else '')
                row_dict['use_cases'].append(
                    row[use_case_col] if use_case_col else '')
            else:
                break

        responses[index] = row_dict

    return responses


def create_multichoice_question_prompt(responses, key_prompt):
    prompts_multichoice = {}

    for idx, data in responses.items():
        prompts_multichoice[idx] = []
        context = data.get('context', '')

        questions = data.get('questions', [])
        answers = data.get('answers', [])
        answers_incorrect = data.get('answers_incorrect', [])
        use_cases = data.get('use_cases', [])

        L = len(questions)

        for i in range(L):
            question = questions[i]
            correct_answer = answers[i] if i < len(answers) else ""
            incorrect_answer = answers_incorrect[i] if i < len(
                answers_incorrect) else ""
            use_case = use_cases[i] if i < len(use_cases) else ""

            prompt = prompt_templates[key_prompt]['template'].format(
                CONTEXT=context, QUESTION=question, ANSWER=correct_answer,
                INCORRECT_ANSWER=incorrect_answer
                )

            prompts_multichoice[idx].append(prompt)

    return prompts_multichoice

def generate_multichoice_questions(prompts_multichoice, model='gpt-4o', seed=0):
    if model != 'gpt-4o':
        raise ValueError("Only 'gpt-4o' model is supported.")

    random.seed(seed)  # Set the random seed for reproducibility

    batch_prompts = []
    idxs = []
    for idx in prompts_multichoice.keys():
        for prompt in prompts_multichoice[idx]:
            idxs.append(idx)
            batch_prompts.append(prompt)

    seeds = [seed] * len(batch_prompts)
    responses = call_gpt_batch(texts=batch_prompts, model=model, seeds=seeds)

    cost = sum([c[1] for c in responses])
    print(f"Cost of llm call ${cost:.3f}")
    msgs = [c[0] for c in responses]

    questions = {}
    for idx, qa in zip(idxs, msgs):
        if idx not in questions.keys():
            questions[idx] = []

        # Assume qa is already a dictionary
        qa_dict = qa

        # Get the choices and the correct answer index
        choices = qa_dict['choices']
        correct_answer_index = qa_dict['answer']

        # Separate the last choice
        last_choice = choices[-1]
        other_choices = choices[:-1]

        # Get the correct answer
        correct_answer = choices[correct_answer_index]

        # Shuffle the other choices
        random.shuffle(other_choices)

        # Reconstruct the choices list with the last choice in its original position
        shuffled_choices = other_choices + [last_choice]

        # Find the new index of the correct answer
        new_correct_answer_index = shuffled_choices.index(correct_answer)

        # Update the qa_dict with the shuffled choices and new answer index
        qa_dict['choices'] = shuffled_choices
        qa_dict['answer'] = new_correct_answer_index

        questions[idx].append(qa_dict)

    return questions


def save_vqa(questions, key_prompt, seed):
    dir_results = dir_path / "generated_questions_text"
    dir_results.mkdir(exist_ok=True, parents=True)
    f_save = dir_results / f"qa_keyprompt_{key_prompt}_seed_{seed}.json"
    with open(f_save, 'w') as f:
        json.dump(questions, f, indent=4)


def add_imgs_to_qa(qas, dir_path, verbose=0):
    """Add the same image or image set to each text question"""
    vqas = copy.deepcopy(qas)

    for idx in vqas.keys():
        dir_images = dir_path / "images" / f"idx_{idx}"
        assert dir_images.exists(), dir_images
        filenames = list(dir_images.iterdir())
        filenames = [f for f in filenames if f.stem[0] != '.']

        try:
            imgs_pil = [Image.open(f).convert('RGB') for f in filenames]
            imgs = [np.array(img) for img in imgs_pil]
            if verbose:
                print(idx, [img.shape for img in imgs])

            for qa in vqas[idx]:
                qa['imgs'] = imgs

        except Exception as e:
            print(f"/nIssue with files {filenames}")
            print(e)

    return vqas


def eval_qa(vqas, key_prompt_eval=0, model='gpt-4o', seed=0):
    """ 
    Run the eval - both with and without the multi-choice options. 
    """

    if model != 'gpt-4o':
        raise ValueError()

    batch_prompts_text = []
    batch_prompts_text_no_choices = []
    batch_prompts_imgs = []
    idxs = []
    question_nums = []
    gts = []

    # make the prompts
    for idx in vqas.keys():
        if 'imgs' not in vqas[idx][0].keys():
            # check the image exists. If not, then exit
            print(f"Skipping idx {idx}, no key `imgs`")
            continue

        for question_num, vqa in enumerate(vqas[idx]):
            batch_prompts_imgs.append(vqa['imgs'])
            idxs.append(idx)
            gts.append(vqa['answer'])
            question_nums.append(question_num)
            
            batch_prompts_text_no_choices.append(vqa['question'])
            # make the version with choices 
            question = vqa['question']
            for num, choice in enumerate(vqa['choices']):
                question += f"\n{num}. {choice}"

            prompt = prompt_eval_templates[key_prompt_eval]['template'].format(QUESTION=question)
            batch_prompts_text.append(prompt)

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
    print(f"Cost of vlm call w choices${cost:.3f}")

    # version where no multichoicing is not happening
    responses_no_choices = call_gpt_batch(texts=batch_prompts_text_no_choices,
                               imgs=batch_prompts_imgs,
                               model=model,
                               json_mode=False,
                               seeds=seeds)
    cost_ = sum([c[1] for c in responses_no_choices])
    print(f"Cost of vlm call w choices${cost_:.3f}")
    
    msgs = [c[0] for c in responses]
    msgs_no_choices = [c[0] for c in responses_no_choices]

    # regex out the predictions
    preds = []
    for msg in msgs:
        pattern = prompt_eval_templates[key_prompt_eval]["regex_pattern"]
        match = re.search(pattern, msg)
        if match is not None: 
            preds.append(int(match.group(1)))
        else:
            preds.append(-1)

    # make `vqas_results` to save the llm responses, and also delete the images
    vqas_results = copy.deepcopy(vqas)
    for i in range(len(idxs)): 
        idx = idxs[i]
        question_num = question_nums[i]

        del vqas_results[idx][question_num]['imgs']
        vqas_results[idx][question_num]['response'] = msgs[i]
        vqas_results[idx][question_num]['response_no_choices'] = msgs_no_choices[i]
        vqas_results[idx][question_num]['pred'] = preds[i]

    gts = np.array(gts)    
    preds = np.array(preds)
    acc = (gts==preds).sum() / len(gts)

    return vqas_results, gts, preds, acc

def save_vqa_results(vqas_results, key_prompt, seed, key_prompt_eval):
    dir_results = dir_path / "generated_questions_text"
    dir_results.mkdir(exist_ok=True, parents=True)
    f_save = dir_results / f"qa_results_keyprompt_{key_prompt}_seed_{seed}_keyprompteval_{key_prompt_eval}.json"
    with open(f_save, 'w') as f:
        json.dump(vqas_results, f, indent=4)

# def save_vqa_results(vqas_results, 


if __name__ == "__main__":
    # config
    key_form = 0
    key_prompt = 1
    seed = 0

    run_eval = 1
    key_prompt_eval = 0
    eval_model = 'gpt-4o'
    eval_seed = 0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = Path(current_dir) / f"formdata_{key_form}"

    # get responses
    responses = get_form_questions(key_form, dir_path)
    # create prompt
    prompts_multichoice = create_multichoice_question_prompt(
        responses, key_prompt)
    # run llm
    qas = generate_multichoice_questions(prompts_multichoice, seed=seed)
    # save

    save_vqa(qas, key_prompt, seed)

    # eval
    if run_eval:
        print(f"running VQA eval")
        vqas = add_imgs_to_qa(qas, dir_path, verbose=0)
        vqas_results, gts, preds, acc = eval_qa(vqas, key_prompt_eval=key_prompt_eval,
            model=eval_model, seed=eval_seed)
        print(f"accuracy {acc}")
        save_vqa_results(vqas_results, key_prompt, seed, key_prompt_eval)

    ipdb.set_trace()
    pass
