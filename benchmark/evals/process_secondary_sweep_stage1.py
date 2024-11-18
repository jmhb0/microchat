"""
python -m ipdb benchmark/evals/process_secondary_sweep_stage1.py
"""
import pandas as pd 
import ipdb
import re

df = pd.read_csv("benchmark/evals/model_prediction_stage1.csv")
ipdb.set_trace()

cols = ['cambrian_13b_prediction', 'VILA1.5-40b_prediction',
       'VILA1.5-13b_prediction', 'cambrian_34b_prediction']
cols_pred = []
df_eval_ = pd.read_csv("benchmark/evals/microsoft-llava-med-v1.5-mistral-7b.csv")
df_eval_['gt'] = df_eval_['gt_answer']
df = df.merge(df_eval_[['key_question', 'gt']], on='key_question')

df_qs = pd.read_csv("benchmark/data/formdata_0/4_questions.csv")
df = df.merge(df_qs[['key_question', 'use_case']], on='key_question')


lookup =  dict(a=0,b=1,c=2,d=3,e=4)
for col in cols:
    # Create new column name by adding 'pred'
    new_col = col.replace('_prediction', '_pred')
    cols_pred.append(new_col)

    # Define function to extract number or return -1
    def extract_number(text):
        try:
            match = re.search(r"answer is \(?([a-e])\)?", text, re.IGNORECASE)
            if match: 
                pred = match.group(1).lower()
                return lookup[pred]
            else:
                return -1
        except:
            return -1

    # Apply extraction to the column and create new column
    df[new_col] = df[col].apply(extract_number)

for col in cols_pred:
    # Create new column name by adding 'pred'
    print()
    acc = f"{(df[col]==df['gt']).mean():.3f}"
    df['correct'] = (df[col]==df['gt'])
    print(df.groupby(['use_case'])['correct'].mean())
    print(acc, col)


# Get the VILA-40B model
if 1: 
    df_vila = df[['key_question', 'gt']].copy()
    df_vila['pred'] = df['VILA1.5-40b_pred']
    df_vila['correct'] = (df_vila['pred']==df_vila['gt'])
    df_vila['msg'] = df['VILA1.5-40b_prediction']
    df_vila.to_csv("~/Downloads/eval_vila15-40b_stage2_prompt0.csv")

ipdb.set_trace()
pass