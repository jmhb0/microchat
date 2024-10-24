"""
python -m ipdb analysis_scripts/20241018_extract_explanations.py
"""

import ipdb
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import logging
import ast
from PIL import Image
import re
from openai import OpenAI

sys.path.insert(0, "..")
sys.path.insert(0, ".")

from models.openai_api import call_gpt_batch
dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)

df = pd.read_csv("/Users/jamesburgess/microchat/benchmark/data/formdata_0/question_strategy_0/df_questions_key_choices_3.csv")

# # original prompt without any kind of system conditions
# prompt_template = """

# """

# batch_prompts_text = []
# for i, row in df.iterrows():
# 	choices = ast.literal_eval(row['choices'])
# 	answer_short = choices['choices'][choices['correct_index']]

# 	prompt = prompt_template.replace("{{question}}", row['question'])
# 	prompt = prompt.replace("{{answer_short}}", answer_short)
# 	prompt = prompt.replace("{{answer_full}}", str(row['answer']))

# 	batch_prompts_text.append(prompt)

# batch_prompts_text = batch_prompts_text[:200]


# overwrite_cache = False
# res = call_gpt_batch(batch_prompts_text,
# 					 imgs=None,
# 					 overwrite_cache=overwrite_cache,
# 					 json_mode=True)
# cost = sum([r[1] for r in res])
# msgs = [m[0]['explanation'] for m in res]




client = OpenAI()

assistant = client.beta.assistants.create(
  name="Math Tutor",
  instructions="You are a personal math tutor. Write and run code to answer math questions.",
  tools=[{"type": "code_interpreter"}],
  model="gpt-4o",
)
thread = client.beta.threads.create()
message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)
from typing_extensions import override
from openai import AssistantEventHandler
 
# First, we create a EventHandler class to define
# how we want to handle the events in the response stream.
 
class EventHandler(AssistantEventHandler):    
  @override
  def on_text_created(self, text) -> None:
    print(f"\nassistant > ", end="", flush=True)
      
  @override
  def on_text_delta(self, delta, snapshot):
    print(delta.value, end="", flush=True)
      
  def on_tool_call_created(self, tool_call):
    print(f"\nassistant > {tool_call.type}\n", flush=True)
  
  def on_tool_call_delta(self, delta, snapshot):
    if delta.type == 'code_interpreter':
      if delta.code_interpreter.input:
        print(delta.code_interpreter.input, end="", flush=True)
      if delta.code_interpreter.outputs:
        print(f"\n\noutput >", flush=True)
        for output in delta.code_interpreter.outputs:
          if output.type == "logs":
            print(f"\n{output.logs}", flush=True)
 
# Then, we use the `stream` SDK helper 
# with the `EventHandler` class to create the Run 
# and stream the response.

with client.beta.threads.runs.stream(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="Please address the user as Jane Doe. The user has a premium account.",
  event_handler=EventHandler(),
) as stream:
  stream.until_done()


ipdb.set_trace()
pass
