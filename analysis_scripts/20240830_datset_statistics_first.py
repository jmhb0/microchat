"""
python analysis_scripts/20240830_datset_statistics_first.py

Summary statistics for a dataset we have. 
Warning: this is the raw questions, where we haven't resolved the "follow up" stuff. 
"""

import ipdb
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

sys.path.insert(0, "..")
sys.path.insert(0, ".")

from models.openai_api import call_gpt_batch

dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)


idx_form = 0
data_src_dir = Path(f"benchmark/build_raw_dataset/formdata_{idx_form}")
df_images = pd.read_csv(data_src_dir / "2_df_images.csv")
df_images = df_images.dropna(subset='caption')
df_questions = pd.read_csv(data_src_dir / "2_df_questions.csv")
df_people = pd.read_csv(data_src_dir / "2_df_people.csv")


def create_bar_chart(modalities, title = None):
    category_counts = Counter(modalities)
    sorted_categories = sorted(category_counts.keys())
    counts = [category_counts[category] for category in sorted_categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_categories, counts, color='black')
    
    # Customize the chart
    ax.set_xlabel('Categories')
    ax.set_ylabel('Count')
    if title: 
    	ax.set_title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    # Adjust layout and display the chart
    plt.tight_layout()
    return fig, ax




###### image based classification
prompt_template = """Below is a text string explaining how an image or set of related images were created. 
Classify the image based on microscopy imaging modality. 
One of: 'light_microscopy', 'fluorescence_microscopy', 'electron_microscopy'. Make your best guess.

Then classify into scale. One of: "tissue", "cellular", "subcellular". 

Return json {'modality' : '...', 'scale' : '...'}

TEXT: 
```
{{text}}
```
"""
batch_prompts = []
for caption in df_images['caption'].values:
	prompt = prompt_template.replace("{{text}}", caption) 
	batch_prompts.append(prompt)

res = call_gpt_batch(batch_prompts, model='gpt-4o', json_mode=True)
msgs = [r[0] for r in res]
modalities = [m['modality'] for m in msgs]
scales = [m['scale'] for m in msgs]

fig, ax = create_bar_chart(modalities, title='(llm-estimated) modalities')
fig.savefig(dir_results / "images_modalities.png")
fig, ax = create_bar_chart(scales, title='(llm-estimated) scales')
fig.savefig(dir_results / "image_scales.png")

# some manual stuff 
image_counts = [str(c) for c in df_images['image_counts'].values]
fig, ax = create_bar_chart(image_counts, title='num-images in the set')
fig.savefig(dir_results / "image_num_imgs_in_set.png")





###### question-based classification
prompt_template = """Below is two strings. 
One is 'context' explaining how an image or set of related images were created. 
One is 'question', which is a question about the images. 

The image fall into one of these types:
	- 'biological', meaning understanding biological featuers of images, understanding biological causes, or designing experiments to understand biology. 
	- 'technical', meaning interpreting the technical aspects of the image like staining or noise, and maybe designing experiments to address technical issues. 
	- 'mixed', it has aspects of both 'biological' and 'technical'

Return json {'type' : '...'}

TEXT: 
```
{{text}}
```

QUESTION: 
```
{{question}}
```
"""
batch_prompts = []
for i, row in df_questions.iterrows():
	question = row['question']
	key_image = row['key_image']
	if key_image not in df_images.index:
		print(f"Warning, missing image {key_image} for question queries")
		continue
	caption = df_images.loc[row['key_image'], 'caption']
	prompt = prompt_template.replace("{{text}}", caption) 
	prompt = prompt.replace("{{question}}", question)
	batch_prompts.append(prompt)

res = call_gpt_batch(batch_prompts, model='gpt-4o', json_mode=True)
msgs = [r[0] for r in res]
types = [m['type'] for m in msgs]

fig, ax = create_bar_chart(types, title='(llm-estimated) types') # biological vs technical
fig.savefig(dir_results / "question_types.png")


## use cases 
use_cases = [str(s) for s in df_questions['use_case'].values]
fig, ax = create_bar_chart([u for u in use_cases if u!='0'], title='(user-defined) use cases')
fig.savefig(dir_results / "question_use_cases.png")

## combine use_cases and question types 
use_cases_cross_types = [f"{u}_{t}" for u, t in zip(use_cases, types)]
fig, ax = create_bar_chart(use_cases_cross_types, title='use cases and types mixed ')
fig.savefig(dir_results / "question_usecases_and_types.png")




ipdb.set_trace()
pass









