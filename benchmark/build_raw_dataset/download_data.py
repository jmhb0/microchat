import ipdb
import requests
from pathlib import Path
import sys
import pandas as pd
import zipfile
import rarfile
import shutil
import tqdm
import re
import os
import filetype
import subprocess
import numpy as np
from urllib.parse import unquote

sys.path.insert(0, '.')
dir_this_file = Path(__file__).parent
urls_form_responses = {
    "0":
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTDny8zzlWTbQ-kdHNu68bl07Pi-HB-JDzHP4PnoA1Q79OkIc9_mLtWYVYkBm4YQ9Te3qZqaRX4df9a/pub?gid=626873631&single=true&output=csv",
}
urls_form_responses_updates = {
    "0":
    "https://docs.google.com/spreadsheets/d/1frCi5IaKlDprEvN0GpSoQOtAJNBQq0lU0ukiN0_vsIA/pub?gid=0&single=true&output=csv",
}


def download_form_responses(idx, url, url_updates, verbose=0):
    dir_data = f"benchmark/data/formdata_{idx}"
    dir_data.mkdir(exist_ok=True, parents=True)

    # download form responses and load
    f_csv = dir_data / "0_responses_raw_download.csv"
    download_csv(url, f_csv)
    df = pd.read_csv(f_csv, dtype=str, keep_default_na=False)

    # download the list of updates
    f_csv_updates = dir_data / "0_edits_responses_raw.csv"
    download_csv(url_updates, f_csv_updates)
    df_updates = pd.read_csv(f_csv_updates, dtype=str, keep_default_na=False)

    # change the column names to more human readable stuff (for the longer names)
    df = change_colnames(df)

    # apply the manual data updates
    df = update_responses(df, df_updates)

    # save the changes
    f_save = dir_data / "1_responses_after_edits0.csv"
    df.to_csv(f_save)

    ipdb.set_trace()

    print("\nUnique emails: ", len(df["Email Address"].unique()))
    print("\nCounts ", df.groupby('Email Address')['Email Address'].count())

    print("Unique names: ", len(df["Your name"].unique()))
    print("\nNames", df.groupby('Your name')['Your name'].count())

    print("downloading the images")
    download_images_from_csv(dir_data, df, verbose=verbose)


def update_responses(df, df_updates):
    """
    Manual adjustments we've made to certain data points.
    Sourced from the sheet in `urls_form_responses_updates`
    """
    df = df.join(df_updates, how='left')
    assert np.array_equal(df['iloc'], df.index.astype(str))
    
    # key is colname of original data, value is the target
    col_mappings = {
        "Email Address" : "Your email",
        "Image / image set": "update_image",
        "Your name": "update_yourname",
        "caption" :  "update_caption",
        "Answer 1" : "update_answer_1", 
        "Question 3" : "update_question_3",
        "Answer 3" : "update_answer_3", 
        "Question 4" : "update_question_4",

    }
    for col_base, col_replace in col_mappings.items():
        col_base_idx = [
            ix for ix, col in enumerate(df.columns) if col == col_base 
        ]
        assert len(col_base_idx) == 1, "col name error"

        col_replace_idx = [
            ix for ix, col in enumerate(df.columns)
            if col == col_replace
        ]
        assert len(col_replace_idx) == 1, "col name error"

        # do the updates if there is content
        mask = (df.iloc[:, col_replace_idx[0]] != '').values
        col_base_name = df.columns[col_base_idx]
        col_replace_name = df.columns[col_replace_idx]
        df.loc[mask, col_base_name] = df.loc[mask, col_replace_name].values

    return df


def change_colnames(df):
    """ 
    The column names all have extra stuff from their descriptions. 
    Let's make the cols more readable
    """

    # for columns with very long names, take their prefix
    df.columns = df.columns.str[:27]

    cols_to_rename = {
        "Image / image set": "Image / image set",
        "Images - sources 1": "Images - source 1",
        "Image - sources 2": "Images source 2",
        "Context - image generation": "Context - image generation",
        "Context: motivation": "Context - motivation",
        "Question 1 use case": "Question 1 use case",
        "Incorrect answer 1 \n": "Incorrect answer 1",
        "Describe the image content ": "caption",
    }

    for prefix, target_name in cols_to_rename.items():
        matching_cols = [col for col in df.columns if col.startswith(prefix)]
        assert len(matching_cols) == 1
        df.rename(columns={matching_cols[0]: target_name}, inplace=True)

    # upadte all the comments
    for i in range(1, 14):
        prefix = f"Comments about question {i}\n"
        target_name = f"Comments about question {i}"
        matching_cols = [col for col in df.columns if col.startswith(prefix)]
        assert len(matching_cols) == 1
        df.rename(columns={matching_cols[0]: target_name}, inplace=True)

    return df


def download_csv(url, output_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print("CSV downloaded and saved at:", output_path)


def determine_file_type(file_path, verbose=0):
    """ for handling different archive file type """
    kind = filetype.guess(file_path)
    if kind is None:
        if verbose:
            print(f"Unknown file type for {file_path}")
        return "unknown"
    if verbose:
        print(f"Detected MIME type for {file_path}: {kind.mime}")
    return kind.mime


def process_archive_file(archive_path, allowed_extensions, verbose=0):
    if verbose:
        print(f"Processing archive: {archive_path}")

    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"The file {archive_path} does not exist.")

    file_size = os.path.getsize(archive_path)
    if verbose:
        print(f"File size: {file_size} bytes")

    if file_size == 0:
        raise RuntimeError("Warning: File is empty.")

    file_type = determine_file_type(archive_path, verbose=verbose)
    parent_dir = archive_path.parent

    if file_type == "application/zip":
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == '/':
                    continue  # skip directories
                zip_info.filename = os.path.basename(zip_info.filename)
                zip_ref.extract(zip_info, parent_dir)
        if verbose:
            print("Extracted ZIP archive")
    elif file_type == "application/x-rar-compressed":
        unar_path = shutil.which('unar')
        if unar_path is None:
            raise RuntimeError(
                "unar command-line tool not found. Please install it using 'brew install unar'."
            )

        # Extract to a temporary directory
        temp_dir = parent_dir / "temp_extract"
        temp_dir.mkdir(exist_ok=True)
        command = [
            unar_path, "-force-overwrite", "-o",
            str(temp_dir),
            str(archive_path)
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to extract RAR archive. Error: {result.stderr}")

        # Check if there's only one directory in temp_dir
        temp_contents = list(temp_dir.iterdir())
        if len(temp_contents) == 1 and temp_contents[0].is_dir():
            # If so, use that as our source
            source_dir = temp_contents[0]
        else:
            # Otherwise, use temp_dir itself
            source_dir = temp_dir

        # Move files from source directory to parent directory
        for item in source_dir.iterdir():
            if item.is_file():
                destination = os.path.join(str(parent_dir),
                                           os.path.basename(str(item)))
                if os.path.exists(destination):
                    os.remove(destination)
                shutil.move(str(item), destination)

        # Remove the temporary directory
        shutil.rmtree(temp_dir)
        if verbose:
            print("Extracted RAR archive")
    else:
        raise ValueError(f"Unsupported archive type: {file_type}")

    # Delete the original archive file
    archive_path.unlink()
    if verbose:
        print(f"Deleted original archive: {archive_path}")

    delete_dotfiles(parent_dir, verbose=verbose)

    # Check for unsupported file types
    for file in parent_dir.iterdir():
        if file.is_file() and file.suffix.lower() not in allowed_extensions:
            print(f"Warning: Unsupported file type found after extraction: {file}")


def extract_file_id(url):
    # for google drive
    file_id_match = re.search(r"(?:\/d\/|id=|open\?id=)([a-zA-Z0-9_-]+)", url)
    return file_id_match.group(1) if file_id_match else None


def delete_dotfiles(directory, verbose=0):
    """Some extracted archive dirs end up with dotfiles, so delete them """
    for file in directory.rglob(".*"):
        try:
            file.unlink()
            if verbose:
                print(f"Deleted dotfile: {file}")
        except Exception as e:
            if verbose:
                print(f"Failed to delete dotfile: {file}, Error: {e}")


def download_images_from_csv(dir_data, df, verbose=0):
    dir_data_images = dir_data / "images"
    dir_data_images.mkdir(exist_ok=True, parents=True)
    col_imgs = [s for s in df.columns if s.lower()[:17] == "image / image set"]
    assert len(
        col_imgs) == 1, "exactly one column must start with name 'image'"
    drive_links = df[col_imgs].iloc[:, 0]
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}

    session = requests.Session()

    for row_index, drive_link in tqdm.tqdm(drive_links.items(),
                                           total=len(drive_links)):
        if row_index <= 107:
            continue
        row_dir = dir_data_images / f"idx_{row_index:04d}"
        row_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Processing link: {drive_link}")

        file_id = extract_file_id(drive_link)
        if not file_id:
            raise ValueError(
                f"Could not extract file ID from link: {drive_link}")

        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        response = session.get(download_url, stream=True)
        response.raise_for_status()

        if "Content-Disposition" in response.headers:
            filename = re.findall("filename=\"(.+)\"",
                                  response.headers["Content-Disposition"])[0]
        else:
            confirm_token = next((value
                                  for key, value in response.cookies.items()
                                  if key.startswith('download_warning')), None)
            if confirm_token:
                download_url = f"{download_url}&confirm={confirm_token}"
                response = session.get(download_url, stream=True)
                response.raise_for_status()
                filename = re.findall(
                    "filename=\"(.+)\"",
                    response.headers["Content-Disposition"])[0]
            else:
                filename = f"file_{row_index}"

        output_path = row_dir / filename
        if verbose:
            print(f"Downloading to: {output_path}")

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if verbose:
            print(f"Downloaded: {filename}")

        file_type = determine_file_type(output_path)
        if file_type in ["application/zip", "application/x-rar-compressed"]:
            process_archive_file(output_path,
                                 allowed_extensions,
                                 verbose=verbose)
        elif output_path.suffix.lower() not in allowed_extensions:
            print(f"Warning: Unsupported file type downloaded: {output_path}")



if __name__ == "__main__":
    verbose = 0
    key = '0'
    download_form_responses(key,
                            urls_form_responses[key],
                            urls_form_responses_updates[key],
                            verbose=verbose)
