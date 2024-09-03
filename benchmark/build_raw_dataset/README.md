### Organization 
In `download_data.csv`, we create data and put it into folder `benchmark/build_raw_dataset/formdata_{idx}` where `idx` is the index for the form number. (So far we only have idx 0). It creates these files. 
- `0_responses_raw_download.csv` is the raw downloaded data sheet having the google form responses.
- `0_edits_responses_raw.csv` is the manual edits to the form responses, which comes from a google sheet. These are corrections to issues, and NOT modifications to the 
- This file will apply the edits, and make `1_responses_after_edits0.csv`. 
- This file also downloads the images into directory `benchmark/build_raw_dataset/formdata_{idx}/images/idx_{num}/` where `num` is the index of the csv file in `1_responses_after_edits0.csv`. So we have one directory per form submission.

In `make_dataframes.csv`
- Reads `1_responses_after_edits0.csv` which has one form submission per row with multiple questions. 
- Creates dataframes `df_people`, `df_questions`, and `df_images`. The images have `key_person` to link pack to the people dataframe, the images have `key_image` and `key_person` to link back to the image and people dataframes respectively. 
- We save lookup dictionaries between things. These ones are 1-1 mappings: `2_lookup_image_to_person.json`, `2_lookup_question_to_person.json`. And we save 1-many mappings (as a list) in `2_lookup_person_to_images.json` and `2_lookup_person_to_questions.json`. 

In `llm_based_annotations.py` we call an LLM to help add metadata annotations or catch errors. These will later be verified. 
- Makes `3_images_updates.csv` which adds a `contexts_updated`. For images rows that are actually multiple images, then if the `context` has references to the filenames, then in replaces those references with a variable like `{idx_0}` or `{idx_1}` (and so on) where the number refers to the index of the image, ordered alphabetically by filename.
- Makes `3_question_updates.csv`. Adds `follow_up` which checks the `'question'` column to see if a question is a follow up question to a prior question. Also adds `'possible_updated_use_case'` which is a bool because the `'use_case'` column is a use case that the LLM predicted to be a different use case. The predicted use case is in `'use_case_llm_predicted'`. Also adds `question_updated` for cases where the question refers to the filename, and it replaces the filename reference to `{idx_0}` or `{idx_1}` (and so on); this is similar to the previous bullet point. 
- The outputs of these are copied into a google sheet, and their responses will help us make manual edits to `df_images` and `df_questions`. We will apply those manual editsin the next step script. 


The 2 csv files from the last script, `3_images_updates.csv` and `3_question_updates.csv` are put to a google sheet, and then we do some manual review on certain content. Those reviews are 
- For `3_images_updates.csv`, we check whether the updates to the context string - for resolving the filenames to `{idx0}`, `{idx1}`, and so on - were correct. We create a column called `contexts_updated_reviewed`, mark it `'X'` if the change is correct, otherwise we manually correct it
- For `3_images_updates.csv`, we .... 
- For `3_question_updates.csv`, we check if the updates to the question string - for resolving the filenames to `{idx0}`, `{idx1}`, and so on - were correct. We create a column called `questions_updated_reviewed`, mark it `'X'` if the change is correct, otherwise we manually correct it

