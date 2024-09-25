import ipdb
import pandas as pd
import sys
from pathlib import Path
import os
import json
import logging
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, '.')
dir_this_file = Path(__file__).parent


def make_dataframes(idx=0):
    """
    The current dataset is one csv of form responses plus a directory of image 
    files. Make separate df's for 
    """
    dir_data = Path(f"benchmark/data/formdata_{idx}")
    f_responses = dir_data / "1_responses_after_edits0.csv"
    df = pd.read_csv(f_responses, dtype=str, keep_default_na=False)

    # assume the index of `df` is the image key
    df_people, lookup_person_to_images, lookup_image_to_person = create_people_dframe(
        df)

    df_images = create_images_dframe(df, dir_data, lookup_image_to_person)

    df_questions, df_people, lookup_person_to_questions, lookup_question_to_person = create_questions_dframe(
        df, df_people, lookup_image_to_person)

    # save dataframes
    df_people.to_csv(dir_data / '2_df_people.csv')
    df_images.to_csv(dir_data / '2_df_images.csv')
    df_questions.to_csv(dir_data / '2_df_questions.csv')

    # save lookups
    with open(dir_data / "2_lookup_person_to_images.json", 'w') as f:
        json.dump(lookup_person_to_images, f, indent=4)
    with open(dir_data / "2_lookup_image_to_person.json", 'w') as f:
        json.dump(lookup_image_to_person, f, indent=4)
    with open(dir_data / "2_lookup_person_to_questions.json", 'w') as f:
        json.dump(lookup_person_to_questions, f, indent=4)
    with open(dir_data / "2_lookup_question_to_person.json", 'w') as f:
        json.dump(lookup_question_to_person, f, indent=4)

    logging.info("Done")

def _map_use_case(use_case):
    """ called in `create_quesitons_dataframe` """
    if pd.isna(use_case) or use_case.strip() == "":
        return 0
    use_case = use_case.lower().strip()
    if "what is unusual or interesting" in use_case:
        return 1
    elif "why am i seeing this" in use_case:
        return 2
    elif "what should we do next" in use_case:
        return 3
    else:
        return 0  # Default to 0 for any other case


def create_questions_dframe(df, df_people, lookup_image_to_person):
    key_images = []
    question_numbers = []
    questions = []
    answers = []
    use_cases = []
    comments = []
    incorrect_answers = []

    # Iterate through the questions
    for i in range(1, 14):
        q_col = f'Question {i}'
        a_col = f'Answer {i}'
        uc_col = f'Question {i} use case'
        c_col = f'Comments about question {i}'
        ia_col = f'Incorrect answer {i}'

        # Check if the question column exists and is not empty
        if q_col in df.columns and not df[q_col].isna().all():
            # Get non-null indices for this question
            indices = df[q_col].dropna().index

            for idx in indices:
                question_text = df.at[idx, q_col]
                # Only add the question if it's not blank
                if pd.notna(question_text) and question_text.strip() != "":
                    key_images.append(idx)
                    question_numbers.append(i)
                    questions.append(question_text)
                    answers.append(df.at[idx, a_col] if a_col in
                                   df.columns else np.nan)
                    use_cases.append(
                        _map_use_case(df.at[idx, uc_col]) if uc_col in
                        df.columns else 0)
                    comments.append(df.at[idx, c_col] if c_col in
                                    df.columns else np.nan)
                    incorrect_answers.append(df.at[idx, ia_col] if c_col in
                                             df.columns else np.nan)

    # Create the new DataFrame
    df_questions = pd.DataFrame({
        'key_image': key_images,
        'question_number': question_numbers,
        'question': questions,
        'answer': answers,
        'use_case': use_cases,
        'comments': comments,
        'incorrect_answer': incorrect_answers,
    })
    df_questions['key_person'] = df_questions['key_image'].map(
        lookup_image_to_person)

    # link to the people df, and also add a question count to the people
    df_questions['key_person'] = df_questions['key_image'].map(
        lookup_image_to_person)

    # create the lookups
    lookup_person_to_questions = df_questions.reset_index().groupby(
        'key_person')['index'].apply(list).to_dict()
    lookup_question_to_person = df_questions['key_person'].to_dict()

    # add a count to the people dataframe
    df_people['num_submitted_questions'] = pd.Series({
        person_id: len(questions)
        for person_id, questions in lookup_person_to_questions.items()
    })

    # Sort the DataFrame by key_image and then by question_number
    df_questions_sorted = df_questions.sort_values(
        ['key_image', 'question_number'])

    df_questions_sorted = df_questions_sorted.reset_index(drop=True).rename_axis('key_question')

    return df_questions_sorted, df_people, lookup_person_to_questions, lookup_question_to_person


def create_images_dframe(df, dir_data, lookup_image_to_person):
    include_prefixes = [
        "Image / image set",
        'Images - source 1',
        'Images source 2',
        'Context - image generation',
        'Context - motivation',
        'caption',
    ]

    def should_include(col):
        return any(col.startswith(prefix) for prefix in include_prefixes)

    columns_to_keep = [col for col in df.columns if should_include(col)]
    df_images = df[columns_to_keep].copy()
    df_images['key_image'] = df_images.index
    df_images.index.name = 'key_image'

    df_images['key_person'] = df_images['key_image'].map(
        lookup_image_to_person).fillna(-1).astype(int)

    df_images['dir_imgs'] = [
        dir_data / f"images/idx_{idx:04d}"
        for idx in df_images['key_image'].values
    ]

    def _get_filenames(directory):
        return sorted([
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ])

    df_images['fnames_images'] = df_images['dir_imgs'].apply(_get_filenames)
    df_images['image_counts'] = [
        len(imgs) for imgs in df_images['fnames_images']
    ]

    return df_images


def create_people_dframe(df):
    include_prefixes = ['Your name', 'Email Address']

    def _should_include(col):
        return any(col.startswith(prefix) for prefix in include_prefixes)

    columns_to_keep = [col for col in df.columns if _should_include(col)]
    df_people = df[columns_to_keep].copy()
    df_people['key_image'] = df_people.index

    # drop rows having no email
    df_people = df_people[df_people['Email Address'] != '']

    # assign a 'person_key' to unique emails
    email_to_key = {
        email: i
        for i, email in enumerate(df_people['Email Address'].unique())
    }
    df_people['key_person'] = df_people['Email Address'].map(email_to_key)

    # identify people with different names and print a warning
    grouped = df_people.groupby('key_person')['Your name'].unique()
    inconsistent_keys = grouped[grouped.apply(len) > 1]
    if len(inconsistent_keys) > 0:
        logging.warning(f"inconsistent keys exist: {inconsistent_keys}")

    # create lookups
    lookup_person_to_images = df_people.groupby('key_person')['key_image'].agg(
        list).to_dict()
    lookup_image_to_person = df_people.set_index(
        'key_image')['key_person'].to_dict()

    # create the final df for people by making 1 row per person
    df_people = df_people.drop_duplicates(subset='key_person', keep='first')
    df_people = df_people.set_index('key_person')

    # cound the form submissions
    df_people['num_submitted_forms'] = df_people.index.map(
        lambda x: len(lookup_person_to_images.get(x, [])))
    df_people = df_people.drop(columns=['key_image'])

    return df_people, lookup_person_to_images, lookup_image_to_person


def create_lookups(df_questions, df_images, df_people):
    lookup_people_to_images

    return lookup_people_to_images, lookup_images_to_questions, lookup_people_to_questions


if __name__ == "__main__":
    verbose = 0
    make_dataframes(idx=0)