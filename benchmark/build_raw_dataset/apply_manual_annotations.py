"""
Apply the annotation updates to all the images and questions. 
Then, for each. 

The convention for applying updates is if 
"""
import ipdb
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from download_data import download_csv

dir_this_file = Path(__file__).parent

urls_question_updates = {
    "0":
    "https://docs.google.com/spreadsheets/d/11d7r2M4OrhwG2ak0mPtnpx73wv5GxPsU6NfPxMk7ydM/pub?gid=821188964&single=true&output=csv",
}
urls_images_updates = {
    "0":
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRt-vRd60QcJj0d15fPVpnUOiIuYDIGxY6p7UtZqu133kU7z6uE4Dj9qSUulk598U_aVPdTJgaTqEFS/pub?gid=1462621761&single=true&output=csv",
}


def update_dataframes(idx):
    dir_data = Path(f"benchmark/data/formdata_{idx}")

    # image updates
    f_csv = dir_data / "4_images_annotated.csv"
    download_csv(urls_images_updates[idx], f_csv)
    df_images = pd.read_csv(f_csv, dtype=str, keep_default_na=False, index_col='key_image')
    map_updates_images = {
        'Context - image generation': 'update_contexts',
    }
    df_images = apply_updates(df_images, map_updates_images)
    df_images = clean_dataframe(df_images)

    # question updates
    f_csv = dir_data / "4_questions_annotated.csv"
    download_csv(urls_question_updates[idx], f_csv)
    df_questions = pd.read_csv(f_csv, dtype=str, keep_default_na=False, index_col='key_question')

    map_updates_questions = {
        # these 3 resolve references to the image number
        "question": 'update_questions_context_references',
        "answer": 'update_answer_context_references',
        "incorrect_answer": 'update_incorrect_answer_context_references',
        # update use cases
        "use_case": 'update_use_case',
        # correct follow up references
        "follow_up": 'update_follow_up',
    }
    df_questions = apply_updates(df_questions, map_updates_questions)
    df_questions = clean_dataframe(df_questions)

    f_csv = dir_data / "4_images.csv"
    df_images.to_csv(f_csv)
    f_csv = dir_data / "4_questions.csv"
    df_questions.to_csv(f_csv)

    return df_images, df_questions


def apply_updates(df, map_updates):
    """ 
    For a dataframe, apply updates from one column to another column. 
    The `map_updates` dict_keys are the col name for the col we will update, and 
    dict_values are the col name holding the updates. 
    The update column is applied only if its text is NOT in ["X", "", "NO"].
    """
    for col_src, col_update in map_updates.items():
        updates = df[col_update].values
        mask = ~np.isin(updates, ["X", "", "NO"])
        df.loc[mask, col_src] = updates[mask]

    return df


def clean_dataframe(df):
    """ remove the cols that were only used for editing annotations etc"""
    drop_cols = [
        col for col in df.columns
        if (col.startswith('update') or col.startswith('Unnamed')
            or col.endswith('updated') or col.startswith('maybe')
            or col.startswith('candidate') or col.startswith('possible') or col.endswith('predicted'))
    ]
    df = df.drop(columns=drop_cols)

    return df


if __name__ == "__main__":
    idx = "0"
    df_images, df_questions = update_dataframes(idx)
    ipdb.set_trace()
    pass
