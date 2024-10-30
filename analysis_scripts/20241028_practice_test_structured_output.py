"""
python analysis_scripts/20241028_practice_test_structured_output.py
"""
import ipdb
import numpy as np
from pydantic import BaseModel
from openai import OpenAI
from textwrap import dedent
from PIL import Image
import base64
import io


client = OpenAI()

class Distractor(BaseModel):
    text: str
    reason: str

class Distractors(BaseModel):
    distractors: list[Distractor]


def base64_to_image(base64_str):
    """
    Convert a base64 string to a PIL Image.
    
    Args:
        base64_str (str): The base64 encoded image string.
        
    Returns:
        PIL.Image.Image: The image object.
    """
    # Decode the base64 string into bytes
    image_data = base64.b64decode(base64_str)
    
    # Convert bytes into a PIL image
    image = Image.open(io.BytesIO(image_data))
    
    return image


def convert_to_multi_choice(item):
    question = item["question"]
    answer = item["answer"]
    image_base64 = item["image"]

    system_prompt = "You are a helpful assistant."
    user_prompt = f"""Please generate 3 distractors for this question given the image:

    Question: {question}
    Answer: {answer}
    """

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": dedent(system_prompt)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": dedent(user_prompt)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            },
        ],
        response_format=Distractors,
    )
    ipdb.set_trace()

    distractors = completion.choices[0].message.parsed.dict()
    choices = [answer] + [distractor["text"] for distractor in distractors["distractors"]]
    reasons = [None] + [distractor["reason"] for distractor in distractors["distractors"]]
    multi_choice_questions = {
        "question": question,
        "choices": choices,
        "reasons": reasons,
        "answer": answer,
    }

    return multi_choice_questions

def _encode_image_np(image_np: np.array):
    """ Encode numpy array image to bytes64 so it can be sent over http """
    assert image_np.ndim == 3 and image_np.shape[-1] == 3
    image = Image.fromarray(image_np)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


if __name__=="__main__":
    random_image = (np.random.rand(128, 128, 3)* 255).astype(np.uint8)
    random_image_base64 = _encode_image_np(random_image)
    item = {
        "question" : "what is in the image?",
        "answer" : "blurry noise",
        "image" : random_image_base64
        }
    convert_to_multi_choice(item)
    ipdb.set_trace()



