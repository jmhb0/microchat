"""
"""
import ipdb
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import random

sys.path.insert(0, '.')
from models.openai_api import call_gpt_batch

dir_this_file = Path(__file__).parent


def gen_choices(key_form, key_question_gen, key_choices_gen, seed=0):
    dir_data_questions = Path(
        f"benchmark/data/formdata_{key_form}/question_strategy_{key_question_gen}"
    )
    df_questions = pd.read_csv(dir_data_questions / "1_df_questions.csv",
                               index_col='key_question')

    if key_choices_gen in (0, 1, 2):
        df_questions = gen_choices_simple(df_questions, key_choices_gen, seed)

    else:
        raise NotImplementedError()

    df_questions['choices'] = make_choices(
        df_questions['llm_response_choices'], seed)

    f_save = dir_data_questions / f"df_questions_key_choices_{key_choices_gen}.csv"
    df_questions.to_csv(f_save)
    print(f"Saved to {f_save}")


prompt_template_simple = {
    # from OmniMedVQA
    0:
    """\
You are an expert in molecular and cell biology, and in microscopy. 
I will give you an original biology-related question and its answer, your task is to rephrase an equivalent question with identical answer. The question related to an image, and we don't show the image.
Meanwhile, I want to transfer this QA-pair into a multi-choice question. Please generate 5 incorrect options to construct the candidate options.
Return a json: {'question' : '...', 'answer' : '...', 'incorrect_answers' : ['...', '...', ...]}

{{question}}
""",
    1:
    """\
You are an expert in molecular and cell biology, and in microscopy. 
I will give you an original biology-related question and its answer, your task is to rephrase an equivalent question with identical answer. The question related to an image, and we don't show the image.
Meanwhile, I want to transfer this QA-pair into a multi-choice question. Please generate 5 incorrect options to construct the candidate options.
I will provide you with one incorrect answer as an example, which you should also rephrase.
Return a json: {'question' : '...', 'answer' : '...', 'incorrect_answers' : ['...', '...', ...], 'sample_incorrect_answer':''}

{{question}}

----
Sample incorrect answer:
'''
{{sample_incorrect_answer}}
'''
"""
}


def gen_choices_simple(df_questions, key_choices_gen, seed):
    """
    The most basic possible llm prompting.  
    """
    if key_choices_gen in (0, 1):
        model = "gpt-4o-mini-2024-07-18 "
        key_prompt_template = key_choices_gen
    elif key_choices_gen == 2:
        model = "gpt-4o-2024-08-06"
        key_prompt_template = 0 # the no incorrect answer
    else:
        raise NotImplementedError()

    batch_prompts = []
    for idx, row in df_questions.iterrows():
        prompt = prompt_template_simple[key_prompt_template]
        prompt = prompt.replace("{{question}}", row['question'])
        # this next line will have no effect for some prompts
        prompt = prompt.replace("{{sample_incorrect_answer}}",
                                str(row['incorrect_answer']))
        batch_prompts.append(prompt)
        assert "{{" not in prompt

    print(f"Running GPT {model} with {len(batch_prompts)} prompts")
    responses = call_gpt_batch(batch_prompts,
                               json_mode=True,
                               model=model,
                               seed=seed)
    cost = sum([c[1] for c in responses])
    print(f"Cost of llm call ${cost:.3f}")
    msgs = [c[0] for c in responses]

    df_questions['llm_response_choices'] = msgs

    return df_questions


def make_choices(questions, seed=0):
    """ 
    Combine the correct and incorrect answer, shuffle them and record the index 
    of the correct answer. 

    Output is a list where each item is this
    {
        "choices": List[str],
        "correct_index": int
    }
    """
    seed_ = seed

    all_choices = []
    for question in questions.values:
        random.seed(seed_)
        choices = question['incorrect_answers'].copy()
        insert_position = random.randint(0, len(choices))
        choices.insert(insert_position, question['answer'])
        random.shuffle(choices)
        correct_index = choices.index(question['answer'])
        all_choices.append(dict(choices=choices, correct_index=correct_index))
        seed_ += 1

    return all_choices


if __name__ == "__main__":
    # which form we collect the quetions from
    key_form = 0
    # which set of questions to get - made in make_questions.py
    key_question_gen = 0
    # key for generating the choices
    key_choices_gen = 2

    gen_choices(key_form, key_question_gen, key_choices_gen, seed=0)
    ipdb.set_trace()
    pass
