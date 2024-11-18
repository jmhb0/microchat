""" 
python -m ipdb benchmark/evals/eval_on_attribute.py
"""
import pandas as pd 
import ipdb
from pathlib import Path

dir_evals = Path("benchmark/graduate_samples/results/run_eval")


df = pd.read_csv("benchmark/data/formdata_0/tagged_nov11_2_bio.csv")

df_gemini = pd.read_csv(dir_evals / "eval_googlegemini-pro-15_stage2_prompt0.csv")
df_gemini = pd.read_csv(dir_evals / "eval_googlegemini-pro-15_stage2_prompt0.csv")
df_gemini['correct_gemini'] = df_gemini['correct']
df = df.merge(df_gemini[['key_question', 'correct_gemini']], on='key_question')

df_llavamed = pd.read_csv("benchmark/evals/microsoft-llava-med-v1.5-mistral-7b.csv")
df_llavamed['correct_llavamed'] = df_llavamed['correct']
df = df.merge(df_llavamed[['key_question', 'correct_llavamed']], on='key_question')


ipdb.set_trace()
pass
print(df.groupby(['_sub_use_case2'])['_sub_use_case2'].count())
print(df.groupby(['modality'])['modality'].count())

print(df['organism'])

