"""
python -m ipdb process_secondary_sweep.py
by YS
"""
import pandas as pd 
import ipdb
import re

df = pd.read_csv("benchmark/evals/model_prediction.csv")
cols = ['Llama-3-VILA1.5-8b_prediction', 'VILA1.5-40b_prediction',
       'VILA1.5-13b_prediction', 'Phi-3.5-Vision_prediction',
       'VILA1.5-3b_prediction', 'idefics2_8b_prediction',
       'cogvlm2-llama3-chat-19B_prediction', 'Phi-3-Vision_prediction']
for col in cols:
	# Create new column name by adding 'pred'
	new_col = col.replace('_prediction', '_pred')

	# Define function to extract number or return -1
	def extract_number(text):
	    try:
	        match = re.search(r"answer is \((\d+)\)", text)
	        return int(match.group(1)) if match else -1
	    except:
	        return -1

	# Apply extraction to the column and create new column
	df[new_col] = df[col].apply(extract_number)

ipdb.set_trace()
pass