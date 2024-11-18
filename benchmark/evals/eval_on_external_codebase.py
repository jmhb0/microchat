""" 
python -m ipdb benchmark/evals/eval_on_external_codebase.py
"""
import pandas as pd 
import ipdb

# llava-med
df_src = pd.read_csv("benchmark/data/formdata_0/4_questions.csv")

def run(df):
	lookup = dict(zip(df_src['key_question'], df_src['use_case']))
	df['use_case'] = [lookup[k] for k in df['key_question']]
	print("*"*80)
	print(df['correct'].mean())
	print(df.groupby(['use_case'])['correct'].mean())

df = pd.read_csv("benchmark/evals/microsoft-llava-med-v1.5-mistral-7b.csv")
run(df)
df = pd.read_csv("benchmark/evals/liuhaotian-llava-v1.6-mistral-7b.csv")
run(df)

ipdb.set_trace()
pass
