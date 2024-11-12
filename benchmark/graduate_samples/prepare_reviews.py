"""
python -m ipdb benchmark/graduate_samples/prepare_reviews.py

"""

import ipdb
from benchmark.refine_bot.run_experiments import _download_csv
# yapf: disable

## which ones where the old reviews? p4. p13. p12. p18
lookup_0 = {
	# sarina 
	"p0" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vSk_MB66ZHDRHnoBiirEN1ex7l7R5iz2TJVk6O1IUpJazik2kR0ZfKrGKgdWq5SDkYxUivwTNxZHYlB/pub?gid=836075084&single=true&output=csv",
	# chad 
	"p4" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vSb7dWY6z56q_WzdaoB2Ys7TS9TDldzMA6DVydQ7gFhBdBd31tfU1FV5Cq0buJfuMnb-EkesB8s5bfL/pub?gid=869104732&single=true&output=csv", # none
	# Zach
	#***** web issue https://docs.google.com/forms/u/1/d/1XfPuLYSMqPly_mkk3wBraGw7hdfsqTIS8z3kxC-Jn8o/edit#responses
	"p6" : "",  
	# Jan 
	"p7" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ16osbeRDh04JtyLou5CPh7_SLY9r9bUbFy1i8GdwMuM-iCNN6niZaAImZW2PpdenqCWER899mUws4/pub?gid=1119612156&single=true&output=csv",
	# Disha
	"p8" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTG3uGArM0Wbfnu3M29au_HjBcuXBi14UGeeMjA1RaFPF7UfQwL81QT8Vf-IfwvkoeMa94_iMf8WZ-y/pub?gid=1516194284&single=true&output=csv",
	# Jesus 
	"p9" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTio1gHmPCCTxSQQ3feGRSPHJK4MuM3mHw5ALcFj388cFg-REnZjVOU766TOrWeEQ62MX5SQ3WTfV5h/pub?gid=21371865&single=true&output=csv",
	# will 
	"p10" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vQVQEOfclpTlR2sAcLu0vm84iB9woXCJ50lfMMiHKKzk0vSlBW_dZoeFjT83l_YoPwLlsAiVaeCMcbp/pub?gid=1596210345&single=true&output=csv",
	# ridhi
	"p12" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vSMoJc6_e37qvwpHaGplq5qoooBRPPOlKjEFConQR_gp94_qMrL-L4lVCjrHKeZvZfvvLp_FJfxUESX/pub?gid=1608002273&single=true&output=csv", # none
	# malvika 
	"p13" :"https://docs.google.com/spreadsheets/d/e/2PACX-1vQ0vvkvfrY8JeS5a26sHhdHz97K0ZW4iL83A2aTi0GQtDyhw4nJwgxOu1cEc7Ps50YRTx_4LnDMjPWc/pub?gid=1366148692&single=true&output=csv", # none
	# alexandra
	"p14" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vSIm3R7jH71fAN1rbsNwYEvZPK2WAID4LU0hYDk98tCMnY2kx0Q_ZJisb-M561BPfpYpZTch2CIJDlD/pub?gid=1764905130&single=true&output=csv",
	# disha - loading issue
	# https://docs.google.com/forms/d/117e6lu-yeycGcxLs8T6VglS-kJBF946v5H6nLVgSxYg/edit#responses
	"p15" : "",	
	# connor 
	"p18" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWurvF1JXemW5WBgZ2kmK16uRNZQRjD6wC1BmtTbe4v8qMjTUWkzZpUZTZm3ttM7_nIddKYp4gLp3t/pub?gid=1258993078&single=true&output=csv", # none
	# zach 
	"p20" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vRrlRuafHh3PkQNu1-r3ZpHc2-w6J_FXv03PexwoE9qgVoVwzQ6a4iLqbfHcwBXPDXYQF92GsWLN8gB/pub?gid=1885341724&single=true&output=csv",
	
}
# yapf: enable

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pandas as pd
import os.path
import pickle

def get_form_responses(form_id):
    """
    Get responses from a Google Form using the Forms API
    
    Parameters:
    form_id (str): The ID of the Google Form
    
    Returns:
    pandas.DataFrame: DataFrame containing the form responses
    """
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/forms.responses.readonly']
    
    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
            
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    try:
        service = build('forms', 'v1', credentials=creds)
        
        # Get form responses
        result = service.forms().responses().list(formId=form_id).execute()
        responses = result.get('responses', [])
        
        # Process responses into a DataFrame
        processed_responses = []
        for response in responses:
            answer_dict = {}
            answers = response.get('answers', {})
            for question_id, answer_data in answers.items():
                # Get the text response (modify this based on your question types)
                answer = answer_data.get('textAnswers', {}).get('answers', [{}])[0].get('value', '')
                answer_dict[question_id] = answer
            processed_responses.append(answer_dict)
        
        return pd.DataFrame(processed_responses)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Example usage:
# Extract form ID from your URL
form_id = "117e6lu-yeycGcxLs8T6VglS-kJBF946v5H6nLVgSxYg"
df = get_form_responses(form_id)
ipdb.set_trace()
pass


