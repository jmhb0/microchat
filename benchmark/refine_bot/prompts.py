"""
"""
from pydantic import BaseModel
import json

# after the eval has been successful, ask the LLM to propose how that's possible.
prompts_reflect = {
    0:
    """\
That is correct. 
Explain how you were able to answer the question without access to the image. What strategies did you use? 
Further, say whether you could deduce the answer with high confidence, or did you make more of a well-informed guess? 
Be concise in your final response.
""",
    1:
    """\
That is correct.

Explain how you were able to answer the question without access to the image - only the question_stem and choices. 
What strategies did you use? 

Then, briefly summarize these "language only strategies" into a short list.

Finally, say whether your reasoning was closer to:
- 'deduction': you answered with high confidence because you had enough information.
- 'abduction': you answered with lower confidence you made an informed guess.

Be concise in your final response.
""",
    2:
    """\
That is correct. Explain how you were able to answer the question. What strategies did you use? 

Then, briefly summarize this into a short list.

Finally, say whether your reasoning was closer to:
- 'deduction': you answered with high confidence because you had enough information.
- 'abduction': you answered with lower confidence you made an informed guess.

Be concise in your final response.
"""
}


class McqQA(BaseModel):
    question_stem: str
    choices: list[str]
    correct_index: int
    explanation: str


# TODO: do include the image caption maybe
prompts_rewrite = {
    0:
    """\
Below, I will display {{n_chat}} chat conversations between a 'user' and an LLM 'assistant'.

In each conversation
	- a user asks the assistant to answer a multichoice VQA question, however they do not provide the image. They only get the question_stem and choices.
	- the question_stem and correct answer are similar to other conversations, however the distractors are different
	- the assistant then answers correctly.
	- the user then asks the assistant to explain how it answered the question with only the text.
	- the assistant then summarizes what stategy they used to answer the question. 
Altogether, these conversations give examples of language shortcuts that need to be avoided when consructing the question_stem and choices.

Here are the conversations:

{{conversations}}


Your task is to rewrite the question_stem and choices so that a different LLM 'assistant' cannot use the language-only strategies that were identified in these past conversations.
Include an 'explanation' about why your new set of distractors are better.
Your revised choices should include the correct answer at the 'correct_index'.

Your revised question_stem and choices should not significantly change the meaning of the question and correct answer.
You are free to change the distractors a lot to achieve this task. 
Include {{n_choices}} choices.
The choices should not be made too different in length and detail to the original question stem and choices.
Note that the question stem may contain important cues that cannot be removed. E.g. if a question asks about the "green stain" in an image, you cannot change it to "stain" because it introduces ambiguity.
The choices should not include the letters in front like (a), (b), ...

Return a json with the following schema: 
""" + json.dumps(McqQA.schema(), indent=2),
    1:
    """\
Below, I will display {{n_chat}} chat conversations between a 'user' and an LLM 'assistant'.

In each conversation
	- a user asks the assistant to answer a multichoice VQA question, however they do not provide the image. They only get the question_stem and choices.
	- the question_stem and correct answer are similar to other conversations, however the distractors are different
	- the assistant then answers correctly.
	- the user then asks the assistant to explain how it answered the question with only the text.
	- the assistant then summarizes what stategy they used to answer the question. 
Altogether, these conversations give examples of language shortcuts that need to be avoided when consructing the question_stem and choices.

CONVERSATIONS:
{{conversations}}


YOUR_TASK:
Your task is to revise the question_stem and choices so that a different LLM 'assistant' cannot use the language-only strategies that were identified in these past conversations.
Include an 'explanation' about why your new set of distractors are better.
Your revised choices should include the correct answer at the 'correct_index'.

CONSTRAINTS_ON_QUESTION_AND_ANSWER:
Your revised question_stem and answer should be semantically equivalent to the original question stem and correct answer.
Your revised question_stem should retain the important visual cues from the original question_stem. E.g. if a question asks about the "green stain" in an image, you should not change it to "stain" because it introduces ambiguity.
Your revised question_stem and answer CAN make certain context-clues more ambiguous, e.g. changing "human cell" to "eukaryotic cell" or "U2Os cell" to "human cell" is fine.
Overall, your revised question_stem and answer CAN make changes to the question and answer overall.
- The original question_stem is "{{question_stem_original}}".
- The original answer is "{{answer_original}}".


OTHER_CONSTRAINTS:
- You are free to change the distractors a lot to achieve your task. 
- The choices should not include the letters in front like (a), (b), ...

OUTPUT_FORMAT
- Include {{n_choices}} choices.
- Return a json with the following schema: 
""" + json.dumps(McqQA.schema(), indent=2),
0:
    """\
Below, I will display {{n_chat}} chat conversations between a 'user' and an LLM 'assistant'.

In each conversation
    - a user asks the assistant to answer a multichoice VQA question, however they do not provide the image. They only get the question_stem and choices.
    - the question_stem and correct answer are similar to other conversations, however the distractors are different
    - the assistant then answers correctly.
    - the user then asks the assistant to explain how it answered the question with only the text.
    - the assistant then summarizes what stategy they used to answer the question. 
Altogether, these conversations give examples of language shortcuts that need to be avoided when consructing the question_stem and choices.

Here are the conversations:

{{conversations}}


Your task is to rewrite the question_stem and choices so that a different LLM 'assistant' cannot use the language-only strategies that were identified in these past conversations.
Include an 'explanation' about why your new set of distractors are better.
Your revised choices should include the correct answer at the 'correct_index'.

Your revised question_stem and choices should not significantly change the meaning of the question and correct answer.
You are free to change the distractors a lot to achieve this task. 
Include {{n_choices}} choices.
The choices should not be made too different in length and detail to the original question stem and choices.
Note that the question stem may contain important cues that cannot be removed. E.g. if a question asks about the "green stain" in an image, you cannot change it to "stain" because it introduces ambiguity.
The choices should not include the letters in front like (a), (b), ...

Return a json with the following schema: 
""" + json.dumps(McqQA.schema(), indent=2),
    2:
    """\
Below, I will display {{n_chat}} chat conversations between a 'user' and an LLM 'assistant'.

In each conversation
    - a user asks the assistant to answer a multichoice VQA question. They only get the question_stem and choices.
    - the question_stem and correct answer are similar to other conversations, however the distractors are different
    - the assistant then answers correctly.
    - the user then asks the assistant to explain the strategies it used.
    - the assistant then summarizes what stategy they used to answer the question. 
Altogether, these conversations give examples of what strategies to avoided when consructing the question_stem and choices.

CONVERSATIONS:
{{conversations}}


YOUR_TASK:
Your task is to revise the question_stem and choices so that a different LLM 'assistant' cannot use the strategies that were identified in these past conversations.
Include an 'explanation' about why your new set of distractors are better.
Your revised choices should include the correct answer at the 'correct_index'.

CONSTRAINTS_ON_QUESTION_AND_ANSWER:
Your revised question_stem and answer should be semantically equivalent to the original question stem and correct answer.
Your revised question_stem should retain the important visual cues from the original question_stem. E.g. if a question asks about the "green stain" in an image, you should not change it to "stain" because it introduces ambiguity.
Your revised question_stem and answer CAN make certain context-clues more ambiguous, e.g. changing "human cell" to "eukaryotic cell" or "U2Os cell" to "human cell" is fine.
Overall, your revised question_stem and answer CAN make changes to the question and answer overall.
- The original question_stem is "{{question_stem_original}}".
- The original answer is "{{answer_original}}".


OTHER_CONSTRAINTS:
- You are free to change the distractors a lot to achieve your task. 
- The choices should not include the letters in front like (a), (b), ...

OUTPUT_FORMAT
- Include {{n_choices}} choices.
- Return a json with the following schema: 
""" + json.dumps(McqQA.schema(), indent=2)
}

prompt_enforce_structure = """\
Below is a response to an LLM query.
It should approximately follow the required response_format. 
Please copy the exact same content, with this reponse_format.

ORIGINAL_RESPONSE: 
{{original_response}}
"""


class PromptCheck(BaseModel):
    is_equivalent: bool
    explanation: str


prompt_check_rewrite = {
    0:
    """\
Below are two question-answer pairs. 
The question-answer pairs are part of VQA triplets, and both pairs use the same image.

Are these question-answer pairs semantically equivalent? 
Or are they significantly different?
Give a true/false and also an explanation.

QUESTION 1:
{{question_stem_1}}

ANSWER 1:
{{answer_1}}

QUESTION 2:
{{question_stem_2}}

ANSWER 2:
{{answer_2}}
""",
    1:
    """\
Below are two question-answer pairs. 
The question-answer pairs are part of VQA triplets, and both pairs use the same image.

Are these question-answer pairs semantically equivalent? 
Or are they significantly different?
Give a true/false and also an explanation.

QUESTION 1:
{{question_stem_1}}

ANSWER 1:
{{answer_1}}

QUESTION 2:
{{question_stem_2}}

ANSWER 2:
{{answer_2}}
"""
}


