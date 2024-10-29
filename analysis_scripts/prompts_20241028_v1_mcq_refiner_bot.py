"""
prompts_20241028_v1_mcq_refiner_bot.py
"""
from pydantic import BaseModel

# after the eval has been successful, ask the LLM to propose how that's possible. 
prompts_reflect = {
    0:
    """\
That is correct. 
Explain how you were able to answer the question without access to the image. What strategies did you use? 
Further, say whether you could deduce the answer with high confidence, or did you make more of a well-informed guess? 
Be concise in your final response.
""", 
	1 : """\
	"""
}

class McqQA(BaseModel):
	question_stem: str
	choices: list[str]
	correct_index: int
	explanation: str

# TODO: do include the image caption maybe
prompts_rewrite = {
	0 : """\
Below, I will display {{n_chat}} chat conversations between a 'user' and an LLM 'assistant'.

In each conversation, a user asks the assistant to answer a multichice VQA question, however they do not provide the image. They only get the question_stem and choices.
The assistant then answers correctly.
The user then asks the assistant to explain how it answered the question with only the text.
The assistant summarizes what stategy they used to answer the question. 


Here are the conversations:

{{conversations}}


Your task is to rewrite the question_stem and choices so that a different LLM 'assistant' cannot use the language-only strategies that were identified in this question without the image. 
You are free to change the distractors a lot to achieve this task. Include an 'explanation' about why your new set of distractors are better.
Your revised choices should leave the correct answer at the same same index in the choices list, called 'correct_index'.

Your revised question_stem and choices should not significantly change the meaning of the question and correct answer.
Note that the question stem may contain important cues that cannot be removed. E.g. if a question asks about the "green stain" in an image, you cannot change it to "stain" because it introduces ambiguity.


Return a json with the following schem

class McqQA(BaseModel):
	question_stem: str
	choices: list[str]
	correct_index: int
	explanation: str
"""
	
}
