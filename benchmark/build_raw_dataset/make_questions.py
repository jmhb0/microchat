"""
make_questions.py
Generate the questions but not the distractors
"""
import ipdb
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from models.openai_api import call_gpt_batch


dir_this_file = Path(__file__).parent

def load_dfs(idx=0):
    dir_data = dir_this_file / f"formdata_{idx}"

    # image updates
    f_csv = dir_data / "4_images.csv"
    df_images = pd.read_csv(f_csv)
    f_csv = dir_data / "4_questions.csv"
    df_questions= pd.read_csv(f_csv)

    return df_images, df_questions

def create_follow_up_chain(df):
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
    
    return df

prompts_sumarize_questions = {
    1 : """\
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
        text = text.replace(placeholder, replacement)
    return text


def construct_question(df_questions, df_images, idx, key_strategy):
    if key_strategy==0:
        raise NotImplementedError()

    prompt_template = prompts_sumarize_questions[key_strategy]

    batch_prompts = []
    for i, row in df_questions.iterrows():
        if i < 120:
            continue 
        if i > 150: 
            break

        context = df_images.loc[row['key_image'], 'Context - image generation']
        question = row['question']
        answer = row['answer']
        prompt = prompt_template.replace("{{context}}", _replace_img_placeholders(context))
        prompt = prompt.replace("{{question}}", _replace_img_placeholders(question))
        prompt = prompt.replace("{{answer}}", _replace_img_placeholders(answer))

        batch_prompts.append(prompt)

    responses = call_gpt_batch(batch_prompts, json_mode=True, model='gpt-4o-mini')
    cost = sum([c[1] for c in responses])
    print(f"Cost of llm call ${cost:.3f}")
    msgs = [c[0] for c in responses]
    cost = []
    ipdb.set_trace()

    pass



if __name__ == "__main__":
    idx = "0"
    key_strategy = 1 
    df_images, df_questions = load_dfs(idx)
    df_questions = create_follow_up_chain(df_questions)
    _ = construct_question(df_questions, df_images, idx, key_strategy)
    ipdb.set_trace()
    pass 




