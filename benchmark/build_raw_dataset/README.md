## Organization
Source links: 
- Form: https://docs.google.com/forms/u/1/d/1t7Z7IIUKrfX_9uaHrA5RtyyadL4DavajLlKA1tecFzE/edit?usp=drive_web
- Google sheet with form responses: https://docs.google.com/spreadsheets/d/1zfNH_d9SqTtxau9aUA5GfTi_WSkwwRv6cUy8LHcKuk4/edit?resourcekey#gid=626873631
- Google sheet for manual form updates: https://docs.google.com/spreadsheets/d/1frCi5IaKlDprEvN0GpSoQOtAJNBQq0lU0ukiN0_vsIA/edit?gid=0#gid=0
- Google sheet for manual review of df_images: 
- Google sheet for manual review of df_questions: 


### 1. download data
In `download_data.csv`, we create data and put it into folder `benchmark/build_raw_dataset/formdata_{idx}` where `idx` is the index for the form number. (So far we only have idx 0). It creates these files. 
- `0_responses_raw_download.csv` is the raw downloaded data sheet having the google form responses.
- `0_edits_responses_raw.csv` is the manual edits to the form responses, which comes from a google sheet. These are corrections to issues, and NOT modifications to the 
- This file will apply the edits, and make `1_responses_after_edits0.csv`. 
- This file also downloads the images into directory `benchmark/build_raw_dataset/formdata_{idx}/images/idx_{num}/` where `num` is the index of the csv file in `1_responses_after_edits0.csv`. So we have one directory per form submission.


### 2. make dataframes for people, images, and questions
In `make_dataframes.py`
- Reads `1_responses_after_edits0.csv` which has one form submission per row with multiple questions. 
- Creates dataframes `df_people`, `df_questions`, and `df_images`. The images have `key_person` to link pack to the people dataframe, the images have `key_image` and `key_person` to link back to the image and people dataframes respectively. 
- We save lookup dictionaries between things. These ones are 1-1 mappings: `2_lookup_image_to_person.json`, `2_lookup_question_to_person.json`. And we save 1-many mappings (as a list) in `2_lookup_person_to_images.json` and `2_lookup_person_to_questions.json`. 

### 2.5 view the downloaded data
BTW, there is also `gen_pdf_initial_responses.py` for putting the downloaded data from `1` into pdfs - one pdf per form submission. 

Some old scripts that I don't use anymore are `gen_pdf_form_responses.py` and `gen_multichoice.py`.
### 3. Use an LLM to do some annotation corrections, and also flag certain data for review
In `llm_based_annotations.py` we call an LLM to help add metadata annotations or catch errors. These will later be verified. 

Make `3_images_updates.csv` which
- adds a `contexts_updated`. For images rows that are actually multiple images, then if the `context` has references to the filenames, then in replaces those references with a variable like `{idx_0}` or `{idx_1}` (and so on) where the number refers to the index of the image, ordered alphabetically by filename.

Make  `3_question_updates.csv` which:
- Similar to inserting the filenames for the image, it does the same for `'question'`, `'answer'`, and `'incorrect_answer'` and proposes corrections in `'question_updated'`, `'answer_updated'`, and `'incorrect_answer_updated'`.
- Questions that are follow ups to prior questions should be marked as such. If it isn't marked, but could be a follow up, it is marked as True in col `'maybe_followup'`. 
- Similarly adds `'possible_updated_use_case'` which is a bool because the `'use_case'` column is a use case that the LLM predicted to be a different use case. The predicted use case is in `'use_case_llm_predicted'`. Also 


### 4. manually do the reviews and then integrate the changes
From the prior scripts, `3_images_updated.csv` and `3_question_updates.csv` are added to a google sheet and we do manual review. The manual review columns have the prefix `'update`. For example, for original column `'question'`, the prior script created `'question_update` with a suggested new value, and so we create a new column `'update_question'`. Either we write `'X'` if we don't want any updates, or we write a string that is the update. 

The script `'apply_manual_annotations.py'` then just applies all those updates, first loading the files `4_images_updated.csv` and `'4_questions_updated.csv` and then saving the updated files to `4_images.csv` and `'4_questions.csv`.

TOOD: updates

