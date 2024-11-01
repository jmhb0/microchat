"""
Usage:
python 20241030_blooms_tagging.py --dataset_name microbench --save_dir /pasteur/u/lmbravo/code/microchat/analysis_scripts/blooms_tagging --send_batch

# to send batch questions
# --send_batch

# flags to continue process
# --batch_id 
# --file_id
"""

import os
import argparse
from datasets import load_dataset
import pandas as pd
import ipdb 
from pydantic import BaseModel
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
    """ + json.dumps(BloomsOutput.schema(), indent=2),
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
    query_qs, all_qs = parse_dataset(args.dataset_name, dataset_info_path)
    
    jsonl_path = os.path.join(ds_save_dir, f'{args.dataset_name}_batch_api.jsonl')
    if not os.path.exists(jsonl_path):
        convert_to_jsonl(query_qs, jsonl_path,
                         prompt_key=args.prompt_key, system_key=args.system_key)
    # 2. call gpt for batch dataset processing
    gpt_output = blooms_tagging(ds_save_dir, jsonl_path,
                                send_batch=args.send_batch,
                                batch_id=args.batch_id,
                                file_id=args.file_id)
    if gpt_output is not None:
        # 4. parse output tags and save to dataset
        # new cols: blooms_question_category, blooms_confidence, blooms_level, blooms_source, blooms_reasoning
        # if there's a counts_dict, then re-organize the output tags from the template questions to the complete dataset
        save_tags(gpt_output, ds_save_dir, query_qs, all_qs=all_qs)

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
    print(f"Processing {dataset_name}...")
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
    # query_qs = pd.DataFrame(query_qs)
    return query_qs, all_qs

def parse_dataset(dataset_name, save_path):
    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        return data['query_qs'], data['all_qs']
    
    if dataset_name == 'microbench':
        query_qs, all_qs = organize_microbench()
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
    example_output = BloomsOutput(blooms_name="example name", blooms_level=1, blooms_reasoning="example reason").dict()
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


def blooms_tagging(save_dir, jsonl_path,
                   send_batch=False, batch_id=None, file_id=None):
    save_path = os.path.join(save_dir, 'gpt_output.jsonl')
    if os.path.exists(save_path):
        print(f"Loading gpt output from {save_path}")
        with open(save_path, 'r') as f:
            gpt_output = json.load(f)
        return gpt_output
    
    if send_batch:
        # 2. call the gpt for blooms tagging
        call_offline_gpt(jsonl_path, save_dir)
    else:
        if batch_id is None or file_id is None:
            raise ValueError("Need to provide a batch_id and file_id to retrieve the output")
        # check if the batch is finished
        status = client.batches.retrieve(batch_id)
        if status == 'completed':
            # 3. retrieve the output
            gpt_output = client.files.content(file_id)
            # 4. save the output
            with open(save_path, 'w') as f:
                json.dump(gpt_output, f)
            return gpt_output
        else:
            raise ValueError(f"Batch job {batch_id} status is {status}")

def plot_histograms(df, column_name, ds_save_dir, bins=20, figsize=(10, 6), title_prefix="Distribution of"):
    """
    Create a histogram for a specified column in the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data
    column_name : str
        Name of the column to create histogram for
    bins : int, default=20
        Number of bins in the histogram
    figsize : tuple, default=(10, 6)
        Size of the figure (width, height)
    title_prefix : str, default="Distribution of"
        Prefix for the plot title
    
    Returns:
    --------
    None (displays the plot)
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create histogram
    sns.histplot(data=df, x=column_name, bins=bins)
    
    # Customize the plot
    plt.title(f"{title_prefix} {column_name}", pad=20)
    plt.xlabel(column_name)
    plt.ylabel("Count")
    
    # Rotate x-axis labels if they're long
    plt.xticks(rotation=45 if df[column_name].dtype == 'object' else 0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(os.path.join(ds_save_dir, 'question_histogram.png'))

def save_tags(gpt_output, save_dir, all_qs=None):
    import ipdb; ipdb.set_trace()
    save_path = os.path.join(save_dir, 'tagged_dataset.csv')
    if os.path.exists():
        print(f"Loading tagged dataset from {save_path}")
        tagged_ds = pd.read_csv(save_path)
    
    gpt_df = pd.DataFrame(gpt_output)
    # make a histogram of the blooms levels in the dataset
    if all_qs is not None:
        blooms_qs = pd.merge(all_qs, gpt_df, on='template_type', how='left')
    else:
        blooms_qs = gpt_df
    # save the tagged dataset
    blooms_qs.to_csv(save_path, index=False)
    # plot the histogram of the blooms levels
    plot_histograms(blooms_qs, 'blooms_level', save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Blooms tags to the dataset')
    parser.add_argument('--dataset_name', type=str, help='dataset name from huggingface', default='microbench')
    parser.add_argument('--save_dir', type=str, help='directory to save the tagged dataset', default='blooms_tagging')
    parser.add_argument('--prompt_key', type=int, help='prompt key to use', default=0)
    parser.add_argument('--system_key', type=int, help='system prompt key to use', default=0)
    parser.add_argument('--send_batch', action='store_true', help='send batch to gpt')
    parser.add_argument('--batch_id', type=str, help='batch id to retrieve the output')
    parser.add_argument('--file_id', type=str, help='file id to retrieve the output')

    args = parser.parse_args()
    main(args)