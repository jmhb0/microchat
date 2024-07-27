import os
import pandas as pd
import ipdb
import re
import json
from pathlib import Path

import sys
sys.path.insert(0, ".")

from models.openai_api import call_gpt_batch

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
            incorrect_answers = answers_incorrect[i] if i < len(
                answers_incorrect) else ""
            use_case = use_cases[i] if i < len(use_cases) else ""

            prompt = prompt_templates[key_prompt]['template'].format(
                CONTEXT=context, QUESTION=question, ANSWER=correct_answer)

            prompts_multichoice[idx].append(prompt)

    return prompts_multichoice


def generate_multichoice_questions(prompts_multichoice, model='gpt-4o', seed=0):
    """ """
    if model != 'gpt-4o':
        raise ValueError()

    batch_prompts = []
    idxs = []
    for idx in prompts_multichoice.keys():
        for prompt in prompts_multichoice[idx]:
            idxs.append(idx)
            batch_prompts.append(prompt)
    
    seeds = [seed]*len(batch_prompts)
    responses = call_gpt_batch(texts=batch_prompts, model=model, seeds=seeds)

    cost = sum([c[1] for c in responses])
    print(f"Cost of llm call ${cost:.3f}")
    msgs = [c[0] for c in responses]

    questions = {}
    for idx, qa in zip(idxs, msgs):
        if idx not in questions.keys():
            questions[idx] = []

        questions[idx].append(qa)

    return questions


if __name__ == "__main__":
    key_form = 0
    key_prompt = 0
    seed = 0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = Path(current_dir) / f"formdata_{key_form}"

    responses = get_form_questions(key_form, dir_path)
    prompts_multichoice = create_multichoice_question_prompt(
        responses, key_prompt)
    questions = generate_multichoice_questions(prompts_multichoice, seed=seed)

    dir_results = dir_path / "generated_questions_text"
    dir_results.mkdir(exist_ok=True, parents=True)
    f_save = dir_results / f"qa_keyprompt_{key_prompt}_seed_{seed}.json"
    with open(f_save, 'w') as f:
        json.dump(questions, f, indent=4)
    ipdb.set_trace()
    pass


