"""
Usage:
python 20241030_blooms_tagging.py --dataset_name ours_nov11_stage2 --version_name 2 --save_dir /pasteur/data/microchat/dataset_versions/nov11/blooms_tagging --dataset_path /pasteur/data/microchat/dataset_versions/nov11/benchmark_nov11_stage2.csv --send_batch
python 20241030_blooms_tagging.py --dataset_name ours_hf_202502 --version_name 3 --save_dir /pasteur/data/microchat/dataset_versions/ours_hf_202502/blooms_tagging --send_batch

python 20241030_blooms_tagging.py --dataset_name llavamed_vqa --save_dir /pasteur/u/minwoos/evaluation/blooms_tagging --dataset_path /pasteur/u/minwoos/evaluation/medical_multimodel_evaluation_data.json --send_batch
python 20241030_blooms_tagging.py --dataset_name mmsci --save_dir /pasteur/u/lmbravo/code/microchat/analysis_scripts/blooms_tagging --dataset_path /pasteur/data/microchat/sota_benchmarks/mmsci/benchmark/test/image_caption_matching_data.json --send_batch
python 20241030_blooms_tagging.py --dataset_name microbench --save_dir /pasteur/u/lmbravo/code/microchat/analysis_scripts/blooms_tagging --send_batch
python 20241030_blooms_tagging.py --dataset_name ours_nov3_s1_naive --save_dir /pasteur/u/lmbravo/code/microchat/analysis_scripts/blooms_tagging --send_batch
python 20241030_blooms_tagging.py --dataset_name omnimed_vqa_part1 --save_dir /pasteur/u/lmbravo/code/microchat/analysis_scripts/blooms_tagging --send_batch
python 20241030_blooms_tagging.py --dataset_name pathchat --save_dir /pasteur/u/lmbravo/code/microchat/analysis_scripts/blooms_tagging --dataset_path /pasteur/data/microchat/sota_benchmarks/PathQABench.json --send_batch
python 20241030_blooms_tagging.py --dataset_name pathvqa --save_dir /pasteur/u/lmbravo/code/microchat/analysis_scripts/blooms_tagging --dataset_path /pasteur/data/microchat/sota_benchmarks/pvqa/qas/test_vqa.pkl --send_batch
python 20241030_blooms_tagging.py --dataset_name lab_bench --save_dir /pasteur/u/lmbravo/code/microchat/analysis_scripts/blooms_tagging --send_batch

# to send batch questions
# --send_batch

# flags to continue process
# --batch_id 
# --file_id
"""

import os
import re
import ast
import argparse
from datasets import load_dataset
import pandas as pd
import pickle
import ipdb 
from pydantic import BaseModel
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from tqdm import tqdm

import openai
from openai import OpenAI
client = OpenAI()

class BloomsOutput(BaseModel):
    blooms_name: str
    blooms_level: int
    blooms_reasoning: str

system_prompts = {
    0: """\
        You are an expert in Biomedical AI with deep knowledge of Bloom's taxonomy and training from the National Board of Medical Examiners. Your role is to assist in designing multiple-choice benchmarks that test vision-language models' perception and reasoning capabilities by classifying questions or question-answer pairs into the most appropriate Bloom's taxonomy level according to the Revised Bloom's taxonomy and NBME guidelines.
        """,
    }

question_prompts = {
    0: """\
    Provide an assessment of the most appropriate Bloom's Taxonomy level for the provided question.
    Question stem:
    {question_stem}
    After the initial evaluation, ask yourself: 'Are you sure about the Bloom's taxonomy category?'
    Double-check your classification and make adjustments if necessary to ensure the question stem accurately reflects the appropriate level of cognitive skills according to Bloom's taxonomy.

    Return a json with the following format:
    {
    blooms_name: str
    blooms_level: int
    blooms_reasoning: str
    }
    """ ,
    1: """\
    Provide an assessment of the most appropriate Bloom's Taxonomy level for the provided question and correct answer pair.
    Question stem:
    {question_stem}
    Answer:
    {correct_answer}

    Return a json with the following format:
    """ + json.dumps(BloomsOutput.schema(), indent=2),
}


def main(args):
    ds_save_dir = os.path.join(args.save_dir, args.dataset_name)
    os.makedirs(ds_save_dir, exist_ok=True)

    # 1. load dataset from huggingface and format into standard format
    dataset_info_path = os.path.join(ds_save_dir, 'dataset_info.npz')
    query_qs, all_qs = parse_dataset(args.dataset_name,
                                     dataset_info_path,
                                     args.dataset_path,
                                     args.version_name)
    
    # 2. call gpt for batch dataset processing
    if args.online_gpt:
        gpt_df = online_blooms_tagging(ds_save_dir, query_qs,
                                           prompt_key=args.prompt_key,
                                           system_key=args.system_key)
        save_tags_online(gpt_df, ds_save_dir, query_qs,
                  all_qs=all_qs,
                  prompt_key=args.prompt_key,
                  system_key=args.system_key)
    else:
        jsonl_path = os.path.join(ds_save_dir, f'{args.dataset_name}_batch_api_{args.system_key}_{args.prompt_key}.jsonl')
        if not os.path.exists(jsonl_path):
            convert_to_jsonl(query_qs, jsonl_path,
                            prompt_key=args.prompt_key, system_key=args.system_key)
        gpt_output = blooms_tagging(ds_save_dir, jsonl_path,
                                send_batch=args.send_batch,
                                batch_id=args.batch_id,
                                prompt_key=args.prompt_key,
                                system_key=args.system_key)
        if gpt_output is not None:
            # 4. parse output tags and save to dataset
            # new cols: blooms_question_category, blooms_confidence, blooms_level, blooms_source, blooms_reasoning
            # if there's a counts_dict, then re-organize the output tags from the template questions to the complete dataset
            save_tags(gpt_output, ds_save_dir, query_qs,
                    all_qs=all_qs,
                    prompt_key=args.prompt_key,
                    system_key=args.system_key)

def save_to_jsonl(data, filename):
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        filename: Name of the file to save to
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return a list of dictionaries."""
    with open(path, buffering=1024*1024) as f:
        return [json.loads(line) for line in f if line.strip()]
    
def organize_ours(file_path=None, dataset_name='ours', version_name=''):
    query_qs = []
    all_qs = None # no need bc we don't have template qs
    if file_path is None:
        # load dataset from huggingface
        ds = load_dataset("jmhb/microvqa", split='train')
        for idx in range(len(ds)):
            row = ds[idx]
            choices = row[f'choices_{version_name}']
            gt_idx = row[f'correct_index_{version_name}']
            correct_answer = choices[gt_idx]
            q_info = {'question_stem': row[f'question_{version_name}'],
            'correct_answer': correct_answer,
            'dataset': dataset_name,
            'id': str(row['key_question']),
            'key_question': str(row['key_question']),
            # 'use_case': str(row['use_case'])
            }
        query_qs.append(q_info)
    else:
        # load dataframe from csv
        df = pd.read_csv(file_path)
        for idx, row in df.iterrows():
            if version_name == '2':
                choices = ast.literal_eval(row[f'choices_{version_name}'])
                gt_idx = row[f'correct_index_{version_name}']
                correct_answer = choices[gt_idx]
            else:
                correct_answer = row[f'answer_{version_name}']
            q_info = {'question_stem': row[f'question_{version_name}'],
                'correct_answer': correct_answer,
                'dataset': dataset_name,
                'id': str(row['key_question']),
                'key_question': str(row['key_question']),
                # 'use_case': str(row['use_case'])
                }
            query_qs.append(q_info)
    return query_qs, all_qs

def organize_microbench():
    """
    Microbench has 2 subdatasets: perception and cognition.
    It also uses question templates, so no need to run blooms on all examples of the template. Also return all_qs for the complete dataset.
    """
    dataset_name = 'microbench'
    all_qs = []
    template_types = []
    query_qs = []
    # perception dataset
    ds = load_dataset("jnirschl/uBench", split='test')
    # ds is a list of dictionaries per image
    # dict_keys(['image_id', 'image', 'label', 'label_name', 'dataset', 'domain', 'institution', 'license', 'microns_per_pixel', 'modality', 'ncbitaxon_id', 'ncbitaxon_name', 'pmid', 'split', 'stain', 'subdomain', 'submodality', 'synthetic', 'captions', 'questions', 'bbox', 'polygon'])
    # get question templates and counts
    for im in tqdm(ds):
        for q_key, q_v  in im['questions'].items():
            if q_v is None:
                continue
            q_info = {'question_stem': q_v['question'],
                'correct_answer': q_v['answer'],
                'dataset': dataset_name,
                'id': q_v['id'],
                'template_type': q_key}
            if q_key not in template_types:
                # init question type
                template_types.append(q_key)
                query_qs.append(q_info)
            all_qs.append(q_info)

    # TODO: add cognition dataset
    return query_qs, all_qs

def organize_omnimed_vqa(file_path, dataset_name='omnimed_vqa_part1'):
    """
    file_path: path to the Open-access folder with the .json files
    Note: has too many qs so splitting into 2 parts
    """
    query_qs = []
    all_qs = None # no need bc we don't have template qs
    q_count = 0
    if 'part1' in dataset_name:
        max_qs = 44000
    else:
        q_count = 44000
        max_qs = 90000 # not real max
    # iterate over subjsons
    for file in os.listdir(file_path):
        if q_count >= max_qs:
            break
        if not file.endswith(".json"):
            continue
        with open(os.path.join(file_path, file)) as f:
            data = json.load(f)
        for q in data:
            if q_count >= max_qs:
                break
            q_info = {'question_stem': q['question'],
                'correct_answer': q['gt_answer'],
                'dataset': dataset_name,
                'id': q['question_id']}
            query_qs.append(q_info)
            q_count += 1
    return query_qs, all_qs

def organize_mmmu_pro(dataset_name='mmmu_pro'):
    ds = load_dataset("MMMU/MMMU_Pro", "standard (4 options)")['test']
    all_letters = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    all_qs = None # no need bc doesn't have template qs
    query_qs = []
    for q in tqdm(ds):
        letter_answer = q['answer']
        options = ast.literal_eval(q['options'])
        # needed bc somethings questions have more than 5 options
        these_letters = all_letters[np.arange(len(options))]
        correct_idx = np.where((these_letters == letter_answer))[0].item()
        correct_answer = options[correct_idx]
        q_info = {'question_stem': q['question'],
            'correct_answer': correct_answer,
            'dataset': dataset_name,
            'id': q['id'],
            'topic_difficulty': q['topic_difficulty'],
            'subject': q['subject']}
        query_qs.append(q_info)
    return query_qs, all_qs


def organize_mmmu(dataset_name='mmmu', set_name='test'):
    subsets =  ['Accounting', 'Agriculture', 'Architecture_and_Engineering',
                'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology',
                'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design',
                'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
                'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
                'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering',
                'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    all_letters = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    for subset in tqdm(subsets):
        ds = load_dataset("MMMU/MMMU", subset)[set_name]
        all_qs = None # no need bc doesn't have template qs
        query_qs = []
        for q in ds:
            if set_name != 'test':
                letter_answer = q['answer']
                options = ast.literal_eval(q['options'])
                # needed bc somethings questions have more than 5 options
                these_letters = all_letters[np.arange(len(options))]
                correct_idx = np.where((these_letters == letter_answer))[0].item()
                correct_answer = options[correct_idx]
            else:
                correct_answer = '?'
            q_info = {'question_stem': q['question'],
                'correct_answer': correct_answer,
                'dataset': dataset_name,
                'id': q['id'],
                'topic_difficulty': q['topic_difficulty'],
                'subject': subset}
            query_qs.append(q_info)
    return query_qs, all_qs

def organize_pathchat(file_path, dataset_name='pathchat'):
    with open(file_path) as f:
        data = json.load(f)
    all_qs = None # no need bc doesn't have template qs
    query_qs = []
    for q in data:
        q_info = {'question_stem': q['question'],
            'correct_answer': '?',
            'dataset': dataset_name,
            'id': str(q['id'])}
        query_qs.append(q_info)
    return query_qs, all_qs

def organize_pathvqa(file_path, dataset_name='pathvqa'):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    all_qs = None # no need bc doesn't have template qs
    query_qs = []
    for q in data:
        q_info = {'question_stem': q['sent'],
            'correct_answer': '?',
            'dataset': dataset_name,
            'id': str(q['question_id']),
            'question_type': q['question_type']}
        query_qs.append(q_info)
    return query_qs, all_qs

def organize_scieval(dataset_name='scieval'):
    ds = load_dataset('OpenDFM/SciEval')['test']
    all_qs = None # no need bc doesn't have template qs
    query_qs = []
    for idx, q in enumerate(ds):
        # TODO: get real answer with the choices
        q_info = {'question_stem': q['question'].split('Which of the following')[0].strip(), # remove the options
            'correct_answer': '?',
            'dataset': dataset_name,
            'id': str(idx)} # q['id] is empty
        query_qs.append(q_info)
    return query_qs, all_qs

def organize_vlm4bio(dataset_name='vlm4bio'):
    subsets = ['Bird', 'Butterfly', 'Fish']
    # TODO: add this one, seems like they have templates so annoying
    # https://github.com/Imageomics/VLM4Bio/blob/main/vlm_datasets/vqa_dataset.py
    pass

def organize_scienceQA(dataset_name='science_qa'):
    all_qs = None # no need bc doesn't have template qs
    query_qs = []
    ds = load_dataset("derek-thomas/ScienceQA")['test']
    for idx, q in enumerate(ds):
        if q['task'] != 'closed choice':
            continue
        correct_answer = q['choices'][q['answer']]
        q_info = {'question_stem': q['question'],
            'correct_answer': correct_answer,
            'dataset': dataset_name,
            'id': f"{idx}_{q['grade']}",
            'grade': q['grade'],
            'skill': q['skill']}
        query_qs.append(q_info)
    return query_qs, all_qs

def organize_mmsci(dataset_path, dataset_name='mmsci'):
    """
    For more info see: https://github.com/Leezekun/MMSci/blob/main/mmsci-data/DATACARD.md#benchmark-data

    """    
    all_qs = None # no need bc doesn't have template qs
    query_qs = []
    with open(dataset_path) as f:
        ds = json.load(f)
    id_num = 0 # making fake id bc test set is annonymized
    # settings: describe the figure, describe the sub-figure, and which subfigure matches this description
    for s_idx, ds_setting in enumerate(ds):
        for q in ds_setting:
            # TODO filter by category to biology here, currently did it by filtering the final dataset.
            # if q['category'] != 'Biological sciences':
            #     continue
            if s_idx == 2:
                pattern = re.compile(r'.*?\?')
                question = pattern.match(q['question']).group()[0]
            else:
                question = q['question']
            id_num += 1
            q_info = {'question_stem': question,
                'correct_answer': '?',
                'dataset': dataset_name,
                'id': str(id_num),
                'category': q['category'],
                'subject': q['subject'],}
            query_qs.append(q_info)
    return query_qs, all_qs

def organize_llavamed_vqa(file_path, dataset_name='llavamed_vqa'):
    """
    file_path: Path to the JSON file containing the custom VQA dataset
    dataset_name: Name of the dataset
    """
    query_qs = []
    all_qs = None  # Not required as there are no template questions

    with open(file_path, 'r') as f:
        data = json.load(f)

    for idx, q in enumerate(data):
        q_info = {
            'question_stem': q['question'].strip(),
            'correct_answer': q['answer'],
            'dataset': q.get('dataset', dataset_name),
            'id': str(idx),
            'subset': q.get('subset', ''),
            'image': q.get('image', []),
            'options': q.get('options', [])
        }
        query_qs.append(q_info)

    return query_qs, all_qs

def extract_core_question(text):
    """
    Efficiently extracts the core question from a scientific query by using
    the last period as a delimiter and minimal processing.
    
    Args:
        text (str): The full text containing background info and the question
        
    Returns:
        str: The extracted core question
    """
    # Split on the last period
    parts = text.rsplit('.', 1)
    
    # If there's no period or only one part, return the cleaned full text
    if len(parts) == 1:
        return parts[0].strip() + ('?' if not parts[0].strip().endswith('?') else '')
    
    # Take the last part and clean it up
    question = parts[-1].strip()
    
    # Add question mark if missing
    if not question.endswith('?'):
        question += '?'
        
    return question

def organize_lab_bench(dataset_name='lab_bench'):
    all_qs = None # no need bc doesn't have template qs
    subsets = ['CloningScenarios', 'DbQA', 'FigQA', 'LitQA2', 'ProtocolQA', 'SeqQA', 'SuppQA', 'TableQA']
    query_qs = []
    for subset in subsets:
        ds = load_dataset("futurehouse/lab-bench", subset)['train']
        for q in ds:
            short_q = q['question']
            if subset == 'CloningScenarios':
                # this subset has long nuclotide sequences as background
                short_q = extract_core_question(q['question'])
            q_info = {'question_stem': short_q,
                'correct_answer': '?',
                'dataset': dataset_name,
                'id': q['id'],
                'subset': subset}
            query_qs.append(q_info)
    return query_qs, all_qs

def parse_dataset(dataset_name, save_path, file_path=None, version_name=''):
    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        return data['query_qs'].tolist(), data['all_qs'].tolist()
    
    print(f"Processing {dataset_name}...")
    if dataset_name == 'microbench':
        query_qs, all_qs = organize_microbench()
    elif 'ours' in dataset_name:
        query_qs, all_qs = organize_ours(file_path, dataset_name,
                                         version_name=version_name)
    elif 'omnimed_vqa' in dataset_name:
        query_qs, all_qs = organize_omnimed_vqa(file_path, dataset_name)
    elif dataset_name == 'mmmu_pro':
        query_qs, all_qs = organize_mmmu_pro(dataset_name)
    elif dataset_name == 'mmmu':
        query_qs, all_qs = organize_mmmu(dataset_name)
    elif dataset_name == 'pathchat':
        query_qs, all_qs = organize_pathchat(file_path, dataset_name)  
    elif dataset_name == 'pathvqa':
        query_qs, all_qs = organize_pathvqa(file_path, dataset_name)
    elif dataset_name == 'scieval':
        query_qs, all_qs = organize_scieval(dataset_name)
    # elif dataset_name == 'vlm4bio':
    #     query_qs, all_qs = organize_vlm4bio(dataset_name)
    elif dataset_name == 'science_qa':
        query_qs, all_qs = organize_scienceQA(dataset_name)
    elif dataset_name == 'lab_bench':
        query_qs, all_qs = organize_lab_bench(dataset_name)
    elif dataset_name == 'mmsci':
        query_qs, all_qs = organize_mmsci(file_path, dataset_name)
    elif dataset_name == 'llavamed_vqa':
        query_qs, all_qs = organize_llavamed_vqa(file_path, dataset_name)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    np.savez(save_path, query_qs=query_qs, all_qs=all_qs)
    return query_qs, all_qs

def update_prompt(prompt, q_info):
    prompt = prompt.replace('{question_stem}', q_info['question_stem'])
    prompt = prompt.replace('{correct_answer}', q_info['correct_answer'])
    return prompt

def convert_to_jsonl(query_qs, save_jsonl, prompt_key=0, system_key=0):
    all_gpt = []
    for q_info in query_qs:
        q_gpt = {"custom_id": q_info['id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "ft:gpt-4o-mini-2024-07-18:marvl-lab:gpt-4o-mini-2024-07-18-blooms:ANR07kPO",
                        "messages": [{"role": "system", "content": system_prompts[system_key]},
                                    {"role": "user", "content": update_prompt(question_prompts[prompt_key], q_info)},
                                    ],
                        "max_tokens": 1000}
                }
        all_gpt.append(q_gpt)
    save_to_jsonl(all_gpt, save_jsonl)

def call_offline_gpt(jsonl_path, save_dir):
    print("Uploading batch input file")
    # upload the batch input file
    batch_input_file = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch"
        )
    # create a batch job
    file_id = batch_input_file.id
    print(f"batch_input_file_id: {file_id}")

    batch_info = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "input_file": os.path.basename(jsonl_path),
        }
    )
    batch_id = batch_info.id
    print(f"batch_id: {batch_id}")
    # save the batch_id and file_id to a txt file
    save_path = os.path.join(save_dir, 'gpt_batch_ids.txt')
    with open(save_path, 'w') as f:
        f.write(f"batch_id: {batch_id}\n file_id: {file_id}")

    # To cancel a batch job:
    # from openai import OpenAI
    # client = OpenAI()
    # client.batches.cancel("batch_abc123")

def call_online_gpt(query_qs, save_path, prompt_key=0, system_key=0):
    responses = []
    if os.path.exists(save_path):
        print(f"Loading gpt output from {save_path}")
        df = pd.read_csv(save_path)
        if len(df) == len(query_qs):
            return df
        else:
            # filter out the ones that are already done
            done_ids = df['id'].values
            query_qs = [q for q in query_qs if int(q['id']) not in done_ids]
            # add the ones that are already done to the responses
            responses = df.to_dict(orient='records')
    
    for q_info in tqdm(query_qs):
        response = client.beta.chat.completions.parse(
            model="ft:gpt-4o-mini-2024-07-18:marvl-lab:gpt-4o-mini-2024-07-18-blooms:ANR07kPO",
            messages=[
                {"role": "system", "content": system_prompts[system_key]},
                {"role": "user", "content": update_prompt(question_prompts[prompt_key], q_info)},
            ],
            max_tokens=1000,
            response_format=BloomsOutput
        )
        output = BloomsOutput.model_validate_json(response.choices[0].message.content).dict()
        output["id"] = q_info['id']
        responses.append(output)
        # save to csv
        df = pd.DataFrame(responses)
        df.to_csv(save_path, index=False)
    return df

def online_blooms_tagging(save_dir, query_qs, prompt_key=0, system_key=0):
    save_path = os.path.join(save_dir, f'gpt_output_{prompt_key}_{system_key}.csv')
    gpt_df = call_online_gpt(query_qs, save_path, prompt_key=prompt_key, system_key=system_key)
    return gpt_df

def blooms_tagging(save_dir, jsonl_path,
                   send_batch=False, batch_id=None,
                   prompt_key=0, system_key=0):
    save_path = os.path.join(save_dir, f'gpt_output_{prompt_key}_{system_key}.jsonl')
    if os.path.exists(save_path):
        print(f"Loading gpt output from {save_path}")
        gpt_output = read_jsonl(save_path)
        return gpt_output

    if send_batch:
        # 2. call the gpt for blooms tagging
        call_offline_gpt(jsonl_path, save_dir)
    else:
        if batch_id is None:
            raise ValueError("Need to provide a batch_id and file_id to retrieve the output")
        # check if the batch is finished
        retreival_info = client.batches.retrieve(batch_id)
        status = retreival_info.status
        output_file_id=retreival_info.output_file_id
        if status == 'completed':
            print(f"Batch job is completed on input_file {retreival_info.metadata['input_file']}")
            # 3. retrieve the output
            gpt_output = client.files.content(output_file_id)
            # 4. save the output
            # TODO: fix writing so we have only the response or clean the response in the next function
            gpt_output.write_to_file(save_path)
            return gpt_output
        elif status == 'failed':
            print(retreival_info)
            ipdb.set_trace()
        else:
            raise ValueError(f"Batch job {batch_id} status is {status}")

def clean_gpt_output(gpt_output):
    final_output = []
    for qs in gpt_output:
        res = qs['response']['body']['choices'][0]['message']['content']
        try:
            res = json.loads(res)
        except json.JSONDecodeError:
            # If parsing fails, create a stripped down version
            match = re.match(r'\s*{\s*"blooms_name":\s*"([^"]+)",\s*"blooms_level":\s*(\d+)', res)
            if match:
                res = {
                    "blooms_name": match.group(1),
                    "blooms_level": int(match.group(2)),
                    "blooms_reasoning": "Error parsing original reasoning"
                }
            else:
                res = {
                    "blooms_name": "Error parsing original name",
                    "blooms_level": -1,
                    "blooms_reasoning": "Error parsing original reasoning"
                }
        res.update({'id': qs['custom_id']})
        final_output.append(res)
    final_df = pd.DataFrame(final_output)
    final_df = final_df[['id', 'blooms_name', 'blooms_level', 'blooms_reasoning']]
    return final_df

def plot_histograms(df, column_name, ds_save_dir, possible_values=None, figsize=(10, 6), title_prefix="Distribution of"):
    """
    Create a histogram for a specified column in the DataFrame, with optional support for
    showing specific possible values even if they don't appear in the data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data
    column_name : str
        Name of the column to create histogram for
    ds_save_dir : str
        Directory to save the plot
    possible_values : list or None, default=None
        List of possible values to include in plot even if not in data
        If None, uses actual data values only
    figsize : tuple, default=(10, 6)
        Size of the figure (width, height)
    title_prefix : str, default="Distribution of"
        Prefix for the plot title
        
    Returns:
    --------
    None (saves the plot)
    """
    # Set the style with improved aesthetics
    plt.style.use('seaborn-v0_8')
    
    # Create figure with higher DPI for better quality
    plt.figure(figsize=figsize, dpi=100)
    
    if possible_values is not None:
        # Convert the column to categorical with specified categories
        temp_series = pd.Categorical(df[column_name], categories=possible_values, ordered=True)
        # Create histogram with categorical data
        ax = sns.histplot(
            data=temp_series,
            discrete=True,
            stat='count',
            color='skyblue',
            edgecolor='navy',
            alpha=0.7
        )
        
        # Ensure all categories are shown
        plt.xticks(range(len(possible_values)), possible_values)
        
    else:
        # Standard histogram for any column type
        ax = sns.histplot(
            data=df,
            x=column_name,
            discrete=True if df[column_name].dtype in ['object', 'category'] else False,
            stat='count',
            color='skyblue',
            edgecolor='navy',
            alpha=0.7
        )
        
        # Handle rotation for string values
        if df[column_name].dtype in ['object', 'category']:
            plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    plt.grid(True, axis='y', alpha=0.3)
    
    # Customize the plot
    plt.title(f"{title_prefix} {column_name}", pad=20, fontsize=12, fontweight='bold')
    plt.xlabel(column_name, fontsize=10)
    plt.ylabel("Count", fontsize=10)
    
    # Add value labels on top of each bar
    for i in ax.patches:
        ax.text(
            i.get_x() + i.get_width()/2,
            i.get_height(),
            int(i.get_height()),
            ha='center',
            va='bottom'
        )
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot with high quality
    plt.savefig(
        os.path.join(ds_save_dir, f'histogram_{column_name}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    
    plt.close()

def save_tags(gpt_output, save_dir, query_qs, all_qs=None,
              prompt_key=0, system_key=0):
    save_path = os.path.join(save_dir, f'blooms_tags_{prompt_key}_{system_key}.csv')
    if os.path.exists(save_path):
        print(f"Loading tagged dataset from {save_path}")
        blooms_qs = pd.read_csv(save_path)
    else:
        gpt_df = clean_gpt_output(gpt_output)
        query_df = pd.DataFrame(query_qs)
        # make a histogram of the blooms levels in the dataset
        if all_qs is not None:
            gpt_df = gpt_df.merge(query_df[['id', 'template_type']], on='id')
            # if there's a template_id it means we used a template to fill in the blooms
            gpt_df.rename(columns={'id': 'template_id'}, inplace=True)
            all_qs_df = pd.DataFrame(all_qs)
            blooms_qs = pd.merge(all_qs_df, gpt_df, on='template_type', how='left')
        else:
            gpt_df = gpt_df.merge(query_df, on='id')
            blooms_qs = gpt_df
        # save the tagged dataset
        blooms_qs.to_csv(save_path, index=False)
    # plot the histogram of the blooms levels
    # remove invalid values
    blooms_qs = blooms_qs[(blooms_qs['blooms_level'] != -1) & (blooms_qs['blooms_level'].notna()) & (blooms_qs['blooms_level'] != 'nan')]
    # make the level a string variable
    blooms_qs['blooms_level'] = blooms_qs['blooms_level'].astype(int).astype(str)
    plot_histograms(blooms_qs, 'blooms_level', save_dir, possible_values=['1', '2', '3', '4', '5', '6'])
    plot_histograms(blooms_qs, 'blooms_name', save_dir)

def save_tags_online(gpt_df, save_dir, query_qs, all_qs=None,
              prompt_key=0, system_key=0):
    save_path = os.path.join(save_dir, f'blooms_tags_{prompt_key}_{system_key}.csv')
    if os.path.exists(save_path):
        print(f"Loading tagged dataset from {save_path}")
        blooms_qs = pd.read_csv(save_path)
    else:
        query_df = pd.DataFrame(query_qs)
        # make a histogram of the blooms levels in the dataset
        if all_qs is not None:
            gpt_df = gpt_df.merge(query_df[['id', 'template_type']], on='id')
            # if there's a template_id it means we used a template to fill in the blooms
            gpt_df.rename(columns={'id': 'template_id'}, inplace=True)
            all_qs_df = pd.DataFrame(all_qs)
            blooms_qs = pd.merge(all_qs_df, gpt_df, on='template_type', how='left')
        else:
            query_df['id'] = query_df['id'].astype(int)
            gpt_df = gpt_df.merge(query_df, on='id')
            blooms_qs = gpt_df
        # save the tagged dataset
        blooms_qs.to_csv(save_path, index=False)
    # plot the histogram of the blooms levels
    # remove invalid values
    blooms_qs = blooms_qs[(blooms_qs['blooms_level'] != -1) & (blooms_qs['blooms_level'].notna()) & (blooms_qs['blooms_level'] != 'nan')]
    # make the level a string variable
    blooms_qs['blooms_level'] = blooms_qs['blooms_level'].astype(int).astype(str)
    plot_histograms(blooms_qs, 'blooms_level', save_dir, possible_values=['1', '2', '3', '4', '5', '6'])
    plot_histograms(blooms_qs, 'blooms_name', save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Blooms tags to the dataset')
    # options: ours, microbench, mmmu, mmmu_pro, omnimed_vqa_part1, omnimed_vqa_part2, pathvqa, pathchat, scieval, science_qa
    parser.add_argument('--dataset_name', type=str, help='dataset name', default='ours')
    parser.add_argument('--save_dir', type=str, help='directory to save the tagged dataset', default='blooms_tagging')
    parser.add_argument('--prompt_key', type=int, help='prompt key to use', default=0)
    parser.add_argument('--system_key', type=int, help='system prompt key to use', default=0)
    parser.add_argument('--send_batch', action='store_true', help='send batch to gpt')
    parser.add_argument('--batch_id', type=str, help='batch id to retrieve the output')
    parser.add_argument('--dataset_path', type=str, help='path to the dataset file', default=None)
    parser.add_argument('--version_name', type=str, help='version name for the dataset. Relevant for ours', default='2')
    parser.add_argument('--online_gpt', action='store_true', help='use online gpt instead of offline')

    args = parser.parse_args()
    main(args)