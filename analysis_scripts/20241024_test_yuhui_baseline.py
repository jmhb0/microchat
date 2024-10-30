"""
python -m ipdb analysis_scripts/20241024_test_yuhui_baseline.py 
"""
import os
from pydantic import BaseModel
from openai import OpenAI
from textwrap import dedent
from PIL import Image
import base64
import pandas as pd
import copy
import io
import ast
from models.openai_api import call_gpt_batch
import re
import numpy as np
from tqdm import tqdm
import argparse


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
    )

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

def path_to_base64(image_path):
    """
    Convert an image file to a base64 string.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        str: The base64 encoded image string.
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_str = base64.b64encode(image_data).decode()
    
    return base64_str


def convert_to_multi_choice(item):
    question = item["question"]
    # answer = item[item["answer"]]
    answer = item["answer"]
    image_base64 = path_to_base64(ast.literal_eval(item["fname_images"])[0])

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

    distractors = completion.choices[0].message.parsed.dict()
    choices = [answer] + [distractor["text"] for distractor in distractors["distractors"]]
    reasons = [None] + [distractor["reason"] for distractor in distractors["distractors"]]
    multi_choice_questions = {
        "mcq_question": question,
        "mcq_choices": choices,
        "mcq_reasons": reasons,
        "mcq_answer": answer,
    }
    return multi_choice_questions

def prepare_data(data_path, remove_context=False):
    df = pd.read_csv(data_path)
    # keep_cols = ["key_question", "question", "answer", "fname_images", "key_image"]
    # df = df[keep_cols]
    if remove_context:
        print("Removing context")
        # look for "Question:" and keep that and everything after
        df["question"] = df["question"].str.split("Question:").str[-1].str.strip()
        df["question"] = "Question: " + df["question"]
    # convert to list of dict form
    # list of dicts, columns are the keys
    q_list = df.to_dict(orient="records")
    
    return q_list

def calculate_acc(preds, labels):
    df = pd.DataFrame({"preds": preds, "labels": labels})
    df["correct"] = df["preds"] == df["labels"]
    acc = df["correct"].mean()
    return acc

def extract_mcs(msgs):
    regex_pattern = r"answer is \(?([a-zA-Z])\)?"
    preds = []
    for msg in msgs:
        match = re.search(regex_pattern, msg)
        if match is not None:
            pred = match.group(1)
            preds.append(pred)
        else:
            preds.append(-1)
    return preds

def build_prompt_get_imgs(df, verbose=False, seed=0, do_shuffle=True):
    PROMPT_SUFFIX = """Think step by step and then output the answer in the format of \"The answer is (X)\" at the end."  """
    PROMPT_PREFIX_NO_IMAGE = """The following question is supposed to be paired with an image. We will not provide the image, so answer to the best of your ability."""

    cache_images = {}
    batch_prompts_imgs = []
    batch_prompts_text = []
    batch_prompts_text_open = []
    batch_prompts_text_no_image = []
    gts = []  # index of the gt

    for idx, row in df.iterrows():
        # get the images
        key_image = row['key_image']
        if key_image in cache_images.keys():
            imgs = cache_images[key_image]
        else:
            filenames = ast.literal_eval(row['fname_images'])
            try:
                imgs_pil = [Image.open(f).convert('RGB') for f in filenames]
                imgs = [np.array(img) for img in imgs_pil]
                cache_images[key_image] = imgs
                if verbose:
                    print(idx, [img.shape for img in imgs])

            except Exception as e:
                print(f"/nIssue with files {filenames}")
                print(e)
                continue

        batch_prompts_imgs.append(imgs)

        ### handle the multiple choice
        choices = row['mcq_choices']
        assert len(choices) in (4, 5, 6)
        # now shuffle
        idxs = np.arange(len(choices))
        np.random.seed(seed + row.name)  # shuffle seed depends on the row
        if do_shuffle:
            idxs = np.random.permutation(idxs)
        choices = [choices[idx] for idx in idxs]
        
        # add letters to it
        str_choices = ""
        letters = ['a', 'b', 'c', 'd', 'e']
        for letter, choice in zip(letters, choices):
            str_choices += f"({letter}) {choice}\n"
        # record the correct answer
        # assuming the correct answer is the first one
        idx_correct = int(np.where(idxs == 0)[0][0]) 
        letter_correct = letters[idx_correct]
        gts.append(letter_correct)

        # save the text prompt
        prompt_text = row['mcq_question'] + "\n\n" + str_choices + PROMPT_SUFFIX
        batch_prompts_text.append(prompt_text)

        # no image prompt
        prompt_text_no_image = PROMPT_PREFIX_NO_IMAGE + prompt_text
        batch_prompts_text_no_image.append(prompt_text_no_image)

        # open prompt
        batch_prompts_text_open.append(row['mcq_question'])

    output = {
        "batch_prompts_text": batch_prompts_text,
        "batch_prompts_imgs": batch_prompts_imgs,
        "gts": gts,
        "batch_prompts_text_no_image": batch_prompts_text_no_image,
        "batch_prompts_text_open": batch_prompts_text_open
    }
    return output

def evaluate(batch_prompts_text, gts, imgs=None, overwrite_cache=True):
    res = call_gpt_batch(batch_prompts_text,
                        imgs=imgs,
                        overwrite_cache=overwrite_cache,
                        json_mode=False)
    msgs = [r[0] for r in res]
    preds = extract_mcs(msgs)
    acc = calculate_acc(preds, gts)
    return acc

def main(question_path, save_path, save_name, remove_context=False, overwrite_cache=False):
    os.makedirs(save_path, exist_ok=True)
    q_list = prepare_data(question_path, remove_context)
    mcq_list = []
    done_list = []
    # load the mcq list if it exists
    if os.path.exists(os.path.join(save_path, save_name)):
        mcq_list = np.load(os.path.join(save_path, save_name), allow_pickle=True).tolist()
        done_list = [mcq["key_question"] for mcq in mcq_list]
    print("Converting to multi choice")
    for item in tqdm(q_list):
        if item["key_question"] in done_list:
            continue
        # check that the image exists and filetype is valid
        filepath = ast.literal_eval(item["fname_images"])[0]
        if not os.path.exists(filepath):
            print(f"Image does not exist for {item['key_question']}")
            continue
        if not filepath.endswith((".jpg", ".jpeg", ".png")):
            extension = filepath.split(".")[-1]
            print(f"Invalid image file type for {item['key_question']}, can't handle {extension} yet")
            continue

        mcq_out = convert_to_multi_choice(item)
        complete_dict = copy.deepcopy(item)
        complete_dict.update(mcq_out)
        mcq_list.append(complete_dict)
        np.save(os.path.join(save_path, save_name), mcq_list)
        done_list.append(item["key_question"])
    mcq_df = pd.DataFrame(mcq_list)
    # for eval on only a couple of questions drop na rows
    mcq_df = mcq_df.dropna()
    print(f"Evaluating on: {len(mcq_df)} questions")
    eval_mcq = build_prompt_get_imgs(mcq_df)
    
    # eval no image
    acc_no_img = evaluate(eval_mcq["batch_prompts_text_no_image"], eval_mcq["gts"], None, overwrite_cache=overwrite_cache)
    print(f"Accuracy no image: {acc_no_img:.2f}")
    # eval with image
    acc_img = evaluate(eval_mcq["batch_prompts_text"], eval_mcq["gts"], eval_mcq["batch_prompts_imgs"], overwrite_cache=overwrite_cache)
    print(f"Accuracy with image: {acc_img:.2f}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--overwrite_cache", action="store_true")
    argparser.add_argument("--remove_context", action="store_true")
    argparser.add_argument("--key_question_gen", type=int, default=0)
    args = argparser.parse_args()
    overwrite_cache = args.overwrite_cache
    remove_context = args.remove_context
    key_question_gen = args.key_question_gen
    # overwrite_cache = False
    # remove_context = False
    # key_question_gen = 0
    context_suffix = "_nocontext" if remove_context else ""
    save_name = f"mcq_list_{key_question_gen}{context_suffix}.npy"
    question_path = f"/pasteur/u/lmbravo/code/microchat/benchmark/data/formdata_0/question_strategy_{key_question_gen}/clean_df_questions.csv"
    save_path = f"/pasteur/u/lmbravo/code/microchat/analysis_scripts/results/20241024_test_yuhui_baseline"
    
    main(question_path, save_path, save_name, remove_context, overwrite_cache)