import ipdb
import sys 

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from models.openai_api import call_gpt

prompt_template = """
I’m creating a dataset to evaluate VLM understanding on biomedical images.
Could you convert this user input question into a multi-choice question with 6 answer choices? 
One choice should be "None of the above", and this choice should have a 1/6 chance of being correct.

Output a JSON format:
{"question": str, "choices": list, "answer": int (start from 0)}.

Context: {CONTEXT}
Input Question: {QUESTION}
Correct Answer: {ANSWER}
"""

questions = [{
	"context" : "Here are fluorescence microscopy images of cells screened with a new drug. The first image is nucleus channel, the second image is cytoskeleton channel.",
	"question" : "Is there anything unusual about this image?",
	"answer" : "this is a multinucleated cell"
}]

idx = 0
prompt_text = prompt_template.replace("{CONTEXT}", questions[idx]["context"])
prompt_text = prompt_text.replace("{QUESTION}", questions[idx]["question"])
prompt_text = prompt_text.replace("{ANSWER}", questions[idx]["answer"])

msg, res = call_gpt(prompt_text, cache=False, json_mode=False)
ipdb.set_trace()
pass