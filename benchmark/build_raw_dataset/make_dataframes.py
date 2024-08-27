import ipdb 
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '.')
dir_this_file = Path(__file__).parent

def make_dataframes(idx=0):
    """
    The current dataset is one csv of form responses plus a directory of image 
    files. Make separate df's for 
    """
    dir_data = dir_this_file / f"formdata_{idx}"
    f_responses = dir_data / "responses_after_updates.csv"
    df = pd.read_csv(f_responses, dtype=str, keep_default_na=False)
    ipdb.set_trace()

    df_questions = create_questions_dframe(df)
    df_images = create_images_dframe(df)
    df_people = create_people_dframe(df)

    # create lookups of keys mapping people->images and images->questions
    lookup_people_to_images, lookup_images_to_quesitons = create_lookups(
        df_questions, df_images, df_people)

def create_questions_dframe(df):
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
                    use_cases.append(df.at[idx, uc_col] if uc_col in
                                     df.columns else np.nan)
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
        'incorrect_answer' : incorrect_answers,
    })

    # Sort the DataFrame by key_image and then by question_number
    df_questions_sorted = df_questions.sort_values(
        ['key_image', 'question_number'])

    return df_questions_sorted


def create_images_dframe(df):
    include_prefixes = [
        "Image / image set", 'Images - source 1', 'Images source 2',
        'Context - image generation', 'Context - motivation', 'caption',
        'Email Address'
    ]

    def should_include(col):
        return any(col.startswith(prefix) for prefix in include_prefixes)

    columns_to_keep = [col for col in df.columns if should_include(col)]
    df_images = df[columns_to_keep].copy()
    df_images['key_form'] = df_images.index  # just to be explicit about it

    return df_images


def create_people_dframe(df):
    ipdb.set_trace()
    include_prefixes = [
        "Image / image set", 'Images - source 1', 'Images source 2',
        'Context - image generation', 'Context - motivation', 'caption',
        'Email Address'
    ]

    def should_include(col):
        return any(col.startswith(prefix) for prefix in include_prefixes)

    columns_to_keep = [col for col in df.columns if should_include(col)]
    df_images = df[columns_to_keep].copy()
    df_images['key_form'] = df_images.index  # just to be explicit about it

    return df_images

 



if __name__ == "__main__":
    verbose = 0
    make_dataframes(idx=0)