## Organization 
These scripts are run after constructing the 'raw vqa dataset' in the folder `benchmark/build_raw_dataset/` - see the README there. 

Those scripts should have created a data folder `benchmark/data/formdata_{key_form}`, where `key_form` is an integer for the particular question collection form we used. 

The final raw images and questions are in `4_df_questions.csv` and `4_df_imges.csv`. The people info for the question creators are in `2_df_people.csv`.

### 1. construct questions
In `make_questions.py`, generate the question and answer text for the VQA. This is the question that will be passed into the standard LLM template. It is saved as `"1_questions.csv"` with columns `'question', 'path_images', 'key_question', 'key_image`.

There are different strategies for generating these questions, indicated by a key, `key_strategy`. 

### 2. generate distractors
In `generate_choices.py`.
