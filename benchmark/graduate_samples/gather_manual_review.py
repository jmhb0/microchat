import os
from tqdm import tqdm
import pandas as pd

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# response_csv_paths = {'nov6': [
#     "https://docs.google.com/spreadsheets/d/e/2PACX-1vSk_MB66ZHDRHnoBiirEN1ex7l7R5iz2TJVk6O1IUpJazik2kR0ZfKrGKgdWq5SDkYxUivwTNxZHYlB/pubhtml?gid=836075084&amp;single=true&amp;widget=true&amp;headers=false",

# ]}

# Step 1: Authenticate and initialize Google Drive
gauth = GoogleAuth()
gauth.LoadCredentialsFile("credentials.json")  # Load saved credentials if they exist

if gauth.credentials is None:
    # Authenticate if credentials are not available
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh if the access token is expired
    gauth.Refresh()
else:
    # Initialize the saved credentials
    gauth.Authorize()

# Save the current credentials to a file
gauth.SaveCredentialsFile("credentials.json")
drive = GoogleDrive(gauth)

def unstack_feedback(feedback_df):
    pass

def get_question_keys(feedback_df):
    pass

def main():
    folder_id_nov5 = "" # nov5
    out_nov5 = compile_responses(folder_id_nov5)
    folder_id_nov6 = "" # nov6
    out_nov6 = compile_responses(folder_id_nov6)

def compile_responses(folder_id, download=True):
    all_out_df = pd.DataFrame()
    # Create local directory to save the downloaded CSV files
    output_dir = "manual_review/responses"
    os.makedirs(output_dir, exist_ok=True)

    subfolder_list = drive.ListFile({'q': f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
    for subfolder in tqdm(subfolder_list):
        subfolder_name = subfolder['title']
        print(f"Accessing subfolder: {subfolder_name}")
        if download:
            download_files_from_subfolder(subfolder['id'], subfolder_name)
        # read the files 
        data_df = pd.read_csv(os.path.join(output_dir, subfolder_name, 'data.csv'))
        feedback_df = pd.read_csv(os.path.join(output_dir, subfolder_name, f"feedback_{subfolder_name} (Responses).csv"))
        feedback_df = unstack_feedback(feedback_df)
        feedback_df = get_question_keys(feedback_df)
        # merge to a df
        out_df = pd.merge(data_df, feedback_df, on=['key_question'], how='left')
        all_out_df = pd.concat([all_out_df, out_df], axis=0)
    # save the df to a csv
    all_out_df.to_csv(os.path.join(output_dir, 'compiled_responses.csv'))

def download_files_from_subfolder(subfolder_id, subfolder_name, output_dir):
    # List all files in the subfolder
    file_list = drive.ListFile({'q': f"'{subfolder_id}' in parents and trashed=false"}).GetList()

    for file in file_list:
        # Download data.csv
        if file['title'] == 'data.csv' and file['mimeType'] == 'text/csv':
            file_name = os.path.join(output_dir, subfolder_name, file['title'])
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            file.GetContentFile(file_name)
            print(f"Downloaded CSV: {file_name}")

        # Download Google Sheets file named "feedback_{subfolder_name} (Responses)"
        elif file['title'] == f"feedback_{subfolder_name} (Responses)" and file['mimeType'] == 'application/vnd.google-apps.spreadsheet':
            file_name = os.path.join(output_dir, subfolder_name, f"{file['title']}.csv")
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            file.GetContentFile(file_name, mimetype='text/csv')
            print(f"Downloaded Google Sheets as CSV: {file_name}")


if __name__ == "__main__":
    main()