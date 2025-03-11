"""
python -m ipdb analysis_scripts/20250309_make_final_dataset.py
Things we need to do:
- Shuffle the choices and the corresponding correct_index
- Add the metadata that Jeff created 
- Apply the sheet with the final corrections which lives at https://docs.google.com/spreadsheets/d/1zwIVpH9vhStjEvytI6S00UxT1Q_o4f6UUmjpAT1nVbY/edit?gid=0#gid=0 
"""
import ipdb
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import datasets

dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)
url_corrections = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQQU2fRXt8ar2Bj-MH_Jyj5n25eO2-VS_NUD0FjGPu12v4apaDVLTCp4V8boPzMt7O2s6a5vlczlkW1/pub?gid=0&single=true&output=csv"
from benchmark.build_raw_dataset.download_data import download_csv

dataset = datasets.load_dataset("jmhb/microvqa_0")['train']

def count_the_correct_index_distribution(dataset):
    correct_index = dataset['correct_index']
    nums, counts = np.unique(correct_index, return_counts=True)
    for num, count in zip(nums, counts):
        print(f"{num}: {count}")


def shuffle_choices_and_correct_index(dataset, random_seed=0):
    random.seed(random_seed)
    
    # Get just the fields we need upfront
    choices_list = dataset['choices']
    correct_indices = dataset['correct_index']
    
    # Pre-allocate lists for new values
    new_choices = []
    new_correct_indices = []
    
    # Process all items first
    for i in tqdm(range(len(dataset)), desc="Shuffling choices and correct indices"):
        choices = choices_list[i]
        correct_index = correct_indices[i]
        correct_answer = choices[correct_index]
        
        # Create shuffled indices
        n_choices = len(choices)
        indices = list(range(n_choices))
        random.shuffle(indices)
        
        # Create new choices list
        shuffled_choices = [choices[idx] for idx in indices]
        
        # Create mapping from old indices to new indices
        old_to_new = {old: new for new, old in enumerate(indices)}
        new_correct_index = old_to_new[correct_index]
        
        new_choices.append(shuffled_choices)
        new_correct_indices.append(new_correct_index)
        
        # Verify correctness (can be removed for production)
        assert shuffled_choices[new_correct_index] == correct_answer
    
    # Update dataset in batches
    dataset = dataset.map(
        lambda x, idx: {
            "choices": [new_choices[i] for i in idx],
            "correct_index": [new_correct_indices[i] for i in idx]
        },
        with_indices=True,
        batched=True,
        batch_size=500 
    )
    
    return dataset

def apply_corrections(dataset: datasets.Dataset):
    download_csv(url_corrections, dir_results / "corrections.csv")
    corrections = pd.read_csv(dir_results / "corrections.csv")
    print(f"Loaded {len(corrections)} corrections")
    
    # Create a dictionary mapping key_question to corrections for efficient lookup
    corrections_dict = {}
    for _, row in corrections.iterrows():
        key_question = row['key_question']
        if key_question not in corrections_dict:
            corrections_dict[key_question] = []
        
        corrections_dict[key_question].append({
            'column': row['column'],
            'old_value': row['old_value'],
            'new_value': row['new_value']
        })
    
    print(f"Organized corrections for {len(corrections_dict)} unique questions")
    
    # Define a simpler, faster function to apply corrections in batches
    def apply_batch_corrections(examples):
        # Only copy the columns we need to modify
        result = {}
        columns_to_modify = set()
        
        # First pass - identify which questions in this batch need corrections
        affected_indices = []
        for i, key_question in enumerate(examples['key_question']):
            if key_question in corrections_dict:
                affected_indices.append(i)
                for correction in corrections_dict[key_question]:
                    columns_to_modify.add(correction['column'])
        
        # Copy only the columns we need to modify
        for column in columns_to_modify:
            if column in examples:
                result[column] = examples[column].copy()
        
        # If no corrections needed for this batch, return empty dict
        if not result:
            return {}
            
        # Second pass - apply corrections only to affected indices
        for i in affected_indices:
            key_question = examples['key_question'][i]
            for correction in corrections_dict[key_question]:
                column = correction['column']
                old_value = correction['old_value']
                new_value = correction['new_value']
                
                # Skip validation for speed, but ensure column exists
                if column not in result:
                    continue
                    
                # Apply correction
                result[column][i] = new_value
        
        return result

    print("Starting correction process...")
    # Apply corrections in batches with smaller batch size
    corrected_dataset = dataset.map(
        apply_batch_corrections,
        batched=True,
        batch_size=100,  # Smaller batch size
        desc="Applying corrections",
        num_proc=1  # Ensure single process for debugging
    )

    return corrected_dataset


# add the metadata 
def load_metadata_1(dataset):
    n_entries = len(dataset)
    metadata_1 = pd.read_csv(dir_results / "benchmark-metadata-mar9.csv")
    # Drop specified columns
    metadata_1 = metadata_1.drop(columns=['key_image', 'key_person'], errors='ignore')
    
    # Convert dataset key_questions to a list for comparison
    dataset_keys = dataset['key_question']
    
    # Sort metadata to match dataset order
    metadata_1 = metadata_1.set_index('key_question').loc[dataset_keys].reset_index()
    assert len(metadata_1) == n_entries

    # Verify the keys match
    assert all(metadata_1['key_question'] == dataset_keys), "key_question mismatch between dataset and metadata"
    
    # Add each column from metadata_1 to the dataset
    for column in metadata_1.columns:
        if column != 'key_question':  # Skip the key column
            dataset = dataset.add_column(column, metadata_1[column].tolist())
    
    return dataset

if 0:
    dataset = load_metadata_1(dataset)
if 1:
    dataset = apply_corrections(dataset)
# do shuffling 
if 1:
    count_the_correct_index_distribution(dataset)
    dataset = shuffle_choices_and_correct_index(dataset, random_seed=0)
    count_the_correct_index_distribution(dataset)
if 1: 
    dataset.push_to_hub('jmhb/microvqa', max_shard_size="500MB")
ipdb.set_trace()
pass




