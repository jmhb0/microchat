"""
python -m ipdb benchmark/make_mcq_benchmark/make_questions.py
Generate the questions but not the distractors. 

We can implement multiple strategies, marked by `key_question_gen`. 
    - `key_question_gen==0`. Do summarize the prior questions for 'followups'. But don't change anything else. 
"""
import ipdb
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import ast

sys.path.insert(0, '.')
from models.openai_api import call_gpt_batch

dir_this_file = Path(__file__).parent


def load_dfs(key_form=0):
    dir_data = Path(f"benchmark/data/formdata_{key_form}")

    # image updates
    f_csv = dir_data / "4_images.csv"
    df_images = pd.read_csv(f_csv, index_col='key_image')
    f_csv = dir_data / "4_questions.csv"
    df_questions = pd.read_csv(f_csv, index_col='key_question')

    return df_images, df_questions


def create_questions(df_questions, df_images, key_question_gen, key_form, add_context=True):
    """ 
    Main function for generating the questions according to the strategy in 
    `key_question_gen`
    """
    dir_data_save = Path(
        f"benchmark/data/formdata_{key_form}/question_strategy_{key_question_gen}"
    )
    dir_data_save.mkdir(exist_ok=True)
    if key_question_gen == 3:
        add_context = False # default True
    
    if key_question_gen in (0, 3):
        df_questions = summarize_follow_up_chain(df_questions,
                                                 df_images,
                                                 key_question_gen=0)
        
        df_questions_final = combine_questions_without_llm(
            df_questions, df_images, add_context=add_context)

    else:
        raise NotImplementedError()

    ipdb.set_trace()
    df_questions_final.to_csv(dir_data_save / f"clean_df_questions.csv")


def combine_questions_without_llm(df_questions, df_images, add_context=True):
    """
    Simplest approach is to just naively create the question by combining the 
    context, question, and (optionally) the prior context from followup questions. 

    Also return the answer.
    """
    df_questions = df_questions.copy()

    questions = []
    fname_images = []
    for i, row in df_questions.iterrows():

        # image information
        row_image = df_images.loc[row['key_image']]
        context = row_image['Context - image generation']
        fs = [
            Path(row_image['dir_imgs']) / f
            for f in ast.literal_eval(row_image['fnames_images'])
        ]
        assert all(f.exists() for f in fs)
        fs = [str(f) for f in fs]

        # create the question string step-by-step
        question = f""
        if add_context:
            question = f"Description of image preparation:\n'''{context}'''\n"
        
        if row['follow_up_chain'] != '0':
            assert row['follow_up_summary'] != ''
            question += f"Additional information:\n'''{row['follow_up_summary']}'''\n"
        question += f"Question:\n'''{row['question']}'''\n"

        # save
        questions.append(question)
        fname_images.append(fs)

    # overwrite the question column, select just the cols we want, and add filenames
    df_questions['question'] = questions
    df_questions['fname_images'] = fname_images
    df_questions = df_questions[[
        'question', 'answer', 'key_image', 'fname_images', 'incorrect_answer',
        'question_number', 'use_case'
    ]]

    return df_questions


def _create_follow_up_chain(df):
    """ 
    called by `summarize_follow_up_chain`

    questions can be followups (indicated by the row 'follow_up'), and that's 
    important, because the question will need to include the information from 
    previous answers. 
    Sometimes the follow-ups might themselves be follups to an older question. 
    Make a new col 'follow_up' in `df_questions` to identify such chains. 
    """

    def build_chain(row):
        chain = []
        current_follow_up = row['follow_up']
        current_key_image = row['key_image']

        for _ in range(15):  # Maximum of 15 iterations
            if current_follow_up == 0:
                break
            assert current_follow_up not in chain, f"Circular reference detected: {current_follow_up} already in chain {chain}"
            chain.append(current_follow_up)
            next_row = df[(df['key_image'] == current_key_image) &
                          (df['question_number'] == current_follow_up)].iloc[0]
            current_follow_up = next_row['follow_up']

        return '-'.join(map(str, reversed(chain))) if chain else 0

    df['follow_up_chain'] = df.apply(build_chain, axis=1)
    df['follow_up_chain'] = df['follow_up_chain'].astype(str)

    return df


prompt_template_summarize_follow_up_chains = {
    0:
    """\
You are an expert in cell and moleculary biology, and microscopy.  
Below I'll give you a string called "context" which explains how some microscopy images were collected. 
Then I'll give you some questions and answers related to that image. If there are multiple questions, they follow on from each other. 

Your task is to summarize the information learned from these questions and answers. 
The response should be statements, and not in the form of questions. Be careful not to remove important information. 
Return a json like this {"summary": "..."}

CONTEXT: 
'''
{{context}}
'''

PAST_QUESTIONS_AND_ANSWERS:
{{past_questions}}
"""
}


def summarize_follow_up_chain(df_questions, df_images, key_question_gen=0):
    if key_question_gen not in prompt_template_summarize_follow_up_chains:
        raise NotImplementedError()

    df_questions = _create_follow_up_chain(df_questions)
    mask = (df_questions['follow_up_chain'] != '0')
    idxs = df_questions[mask].index

    batch_prompts = []

    for idx, row in df_questions.loc[idxs].iterrows():
        past_questions = row['follow_up_chain'].split("-")
        assert len(past_questions) > 0
        rows_this_image = df_questions[df_questions['key_image'] ==
                                       row['key_image']].copy()
        rows_this_image['question_number'] = rows_this_image[
            'question_number'].astype(str)
        rows_this_image = rows_this_image.set_index('question_number')

        rows_past_questions = rows_this_image.loc[past_questions]
        assert len(rows_past_questions) == len(past_questions)

        past_questions_str = ""
        for i, (idx_, row_) in enumerate(rows_past_questions.iterrows()):
            past_questions_str += f"question {i+1}:\n '''{row_['question']}''' \n"
            past_questions_str += f"answer {i+1}:\n '''{row_['answer']}''' \n"
            past_questions_str += "-" * 10 + "\n"

        context = df_images.loc[row['key_image'], 'Context - image generation']

        prompt = prompt_template_summarize_follow_up_chains[key_question_gen]
        prompt = prompt.replace("{{context}}", context)
        prompt = prompt.replace("{{past_questions}}", past_questions_str)
        batch_prompts.append(prompt)

    responses = call_gpt_batch(batch_prompts,
                               json_mode=True,
                               model='gpt-4o-2024-08-06')
                            #    model='gpt-4o-mini')
    cost = sum([c[1] for c in responses])
    print(f"Cost of llm call ${cost:.3f}")
    msgs = [c[0] for c in responses]

    df_questions['follow_up_summary'] = ""
    df_questions.loc[idxs, 'follow_up_summary'] = [m['summary'] for m in msgs]

    return df_questions


# prompt used by function `construct_question`
prompts_sumarize_questions = {
    # this prompt didn't work too well. It tended to to remove essential info from the context. For now, not going to summarize the question
    0:
    """\
I’m creating a dataset to evaluate VLM understanding on microscopy images.

Below is the 'context', which explains how the image was prepared. 
Then there's a 'question'. 

Then there's an answer. The 'answer' may contain a longer 'rationale'

Combine the 'context' and 'question' to create a new 'question_' that is concise and clear, but does not discard important information needed for the answer. 

The 'answer' might contain both an answer and a detailed rationale. 
Return 'answer_' and 'rationale_'. If the 'answer' does not have a rationale, then make 'rationale_' empty. 
The 'answer_' and 'rationale_' should also be concise and clear, but not discard important information.

For all responses, if there is reference to 'image 1' or 'image 2' and so on, then that should be retained.

Return a json, {'question_':'...', 'answer_':'...', 'rationale_':'...',}


CONTEXT: 
'''
{{context}}
'''
QUESTION: 
'''
{{question}}
'''
ANSWER: 
'''
{{answer}}
'''
"""
}


def _replace_img_placeholders(text):
    """ Refine the image placeholders """
    for i in range(9):
        placeholder = f"{{img{i}}}"
        replacement = f"image {i}"
        text = str(text).replace(placeholder, replacement)
    return text


def bak_construct_question(df_questions, df_images, idx, key_question_gen):
    """ 
    for now, not using this function bc it removed too much info with the prompt 
    I tried, and it's not too necessary anyways 
    """
    if key_question_gen == 0:
        raise NotImplementedError()

    prompt_template = prompts_sumarize_questions[key_question_gen]

    batch_prompts = []
    for i, row in df_questions.iterrows():
        context = df_images.loc[row['key_image'], 'Context - image generation']
        question = row['question']
        answer = row['answer']
        prompt = prompt_template.replace("{{context}}",
                                         _replace_img_placeholders(context))
        prompt = prompt.replace("{{question}}",
                                _replace_img_placeholders(question))
        prompt = prompt.replace("{{answer}}",
                                _replace_img_placeholders(answer))

        batch_prompts.append(prompt)

    responses = call_gpt_batch(batch_prompts,
                               json_mode=True,
                               model='gpt-4o-mini')
    cost = sum([c[1] for c in responses])
    print(f"Cost of llm call ${cost:.3f}")
    msgs = [c[0] for c in responses]
    cost = []

    pass


# prompt used by function `construct_question`
prompts_breakout_answer_and_rationale = {
    0:
    """\
You are an expert in cell and moleculary biology, and microscopy. 
I will give you a 'context' string which explains how a microscopy image was prepared. 
I will also give you a 'question' and 'answer' string. 

If the 'answer' is long, then it may contain a longer 'explanation' for why the answer is correct. 
It may also contain 'discussion' that elaborates on the answer to highlight the importance of the question. 


Your task is to notice if the answer is long, and if it is, then separate the 'answer' string into 'answer_no_explanation', 'explanation', and 'discussion'. 
If the 'answer' does not contain an 'explanation' or a 'discusison', then return empty string for those fields.

Return a json, {'answer':'...', 'answer_no_explanation':'...', 'explanation':'...', 'discussion':'...'}


CONTEXT: 
'''
{{context}}
'''
QUESTION: 
'''
{{question}}
'''
ANSWER: 
'''
{{answer}}
'''
"""
}


def answer_breakout(df_questions, df_images, idx, key_question_gen):
    """ 
    break out the answer into 'explanation' and 'discussion'
    """

    prompt_template = prompts_breakout_answer_and_rationale[key_question_gen]

    batch_prompts = []
    for i, row in df_questions.iterrows():
        if i < 120:
            continue
        if i > 250:
            break

        context = df_images.loc[row['key_image'], 'Context - image generation']
        question = row['question']
        answer = row['answer']
        prompt = prompt_template.replace("{{context}}",
                                         _replace_img_placeholders(context))
        prompt = prompt.replace("{{question}}",
                                _replace_img_placeholders(question))
        prompt = prompt.replace("{{answer}}",
                                _replace_img_placeholders(answer))

        batch_prompts.append(prompt)

    responses = call_gpt_batch(batch_prompts,
                               json_mode=True,
                               model='gpt-4o-mini')
    cost = sum([c[1] for c in responses])
    print(f"Cost of llm call ${cost:.3f}")
    msgs = [c[0] for c in responses]


if __name__ == "__main__":
    key_form = 0
    key_question_gen = 3 # 0 default, 3 is no context in final question
    df_images, df_questions = load_dfs(key_form)
    create_questions(df_questions, df_images, key_question_gen, key_form)
    ipdb.set_trace()
    pass
