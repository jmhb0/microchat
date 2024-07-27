"""
Functions for calling api
Needs to have set OPENAI_API_KEY.
Models: https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
"""
import ipdb
import base64
import asyncio
from openai import OpenAI, AsyncOpenAI, ChatCompletion
from pathlib import Path
from PIL import Image
import numpy as np
import os
import io
import shutil
from typing import List
import lmdb
import json
import sys
import logging
import concurrent.futures
from threading import Lock
import time
import tqdm

import sys

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from cache import cache_utils

client = OpenAI()
cache_openai = lmdb.open("cache/cache_openai", map_size=int(1e11))
cache_lock = Lock()


def call_gpt(
    # args for setting the `messages` param
    text: str,
    imgs: List[np.ndarray] = None,
    system_prompt: str = None,
    json_mode: bool = True,
    # kwargs for client.chat.completions.create
    detail: str = "high",
    model: str = "gpt-4o",
    temperature: float = 1,
    max_tokens: int = 2048,
    top_p: float = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    seed: int = 0,
    n: int = 1,
    # args for caching behaviour
    cache: bool = True,
    overwrite_cache: bool = False,
    num_retries:
    # if json_mode=True, and not json decodable, retry this many time
    int = 3):
    """ 
    Call GPT LLM or VLM synchronously with caching.
    To call this in a batch efficiently, see func `call_gpt_batch`.

    If `cache=True`, then look in database ./cache/cache_openai for these exact
    calling args/kwargs. The caching only saves the first return message, and not
    the whole response object. 

    imgs: optionally add images. Must be a sequence of numpy arrays. 
    overwrite_cache (bool): do NOT get response from cache but DO save it to cache.
    seed (int): doesnt actually work with openai API atm, but it is in the 
        cache key, so changing it will force the API to be called again
    """
    # response format
    if json_mode:
        response_format = {"type": "json_object"}
    else:
        response_format = {"type": "text"}

    # system prompt
    messages = [{
        "role": "system",
        "content": system_prompt,
    }] if system_prompt is not None else []

    # text prompt
    content = [
        {
            "type": "text",
            "text": text,
        },
    ]

    # for imgs, put a hash key representation in content for now. If not cahcing,
    # we'll replace this value later (it's because `_encode_image_np` is slow)
    if imgs:
        content.append(
            {"imgs_hash_key": [cache_utils.hash_array(im) for im in imgs]})

    # text & imgs to message - assume one message only
    messages.append({"role": "user", "content": content})

    # arguments to the call for client.chat.completions.create
    kwargs = dict(
        messages=messages,
        response_format=response_format,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        n=n,
    )

    if cache:
        cache_key = json.dumps(kwargs, sort_keys=True)
        with cache_lock:
            msg = cache_utils.get_from_cache(cache_key, cache_openai)
        if msg is not None and not overwrite_cache:
            if json_mode:
                msg = json.loads(msg)
            return msg, None

    # not caching, so if imgs,then encode the image to the http payload
    if imgs:
        assert "imgs_hash_key" in content[-1].keys()
        content.pop()

        base64_imgs = [_encode_image_np(im) for im in imgs]
        for base64_img in base64_imgs:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    "detail": detail,
                },
            })

    # call gpt if not cached. If json_mode=True, check that it's json and retry if not
    for i in range(num_retries):

        response = client.chat.completions.create(**kwargs)
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        msg = response.choices[0].message.content

        if json_mode:
            try:
                _ = json.loads(msg)
                break  # successfully output json

            except json.decoder.JSONDecodeError:
                if i == num_retries - 1:
                    raise f"Response not valid json, after {num_retries} tries"
                logging.info(f"Response not valid json, retrying")
                continue

    # save to cache if enabled
    if cache:
        with cache_lock:
            cache_utils.save_to_cache(cache_key, msg, cache_openai)

    if json_mode:
        msg = json.loads(msg)

    response = dict(prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens)

    return msg, response


def _encode_image_np(image_np: np.array):
    """ Encode numpy array image to bytes64 so it can be sent over http """
    assert image_np.ndim == 3 and image_np.shape[-1] == 3
    image = Image.fromarray(image_np)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def call_gpt_batch(texts,
                   imgs=None,
                   seeds=None,
                   json_modes=None,
                   get_meta=True,
                   **kwargs):
    """ 
    with multithreading
    if return_meta, then return a dict that tells you the runtime, the cost
    """
    n = len(texts)
    if imgs is None:
        imgs = [None] * n

    assert n == len(imgs), "texts and imgs must have the same length"

    # handle having a different seed per call
    all_kwargs = [kwargs.copy() for _ in range(n)]
    if seeds is not None or json_modes is not None:
        for i in range(n):
            if seeds is not None:
                all_kwargs[i]['seed'] = seeds[i]
            if json_modes is not None:
                all_kwargs[i]['json_mode'] = json_modes[i]

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = []

        for text, img, _kwargs in zip(texts, imgs, all_kwargs):
            future = executor.submit(call_gpt, text, img, **_kwargs)
            futures.append(future)

        # run
        results = [list(future.result()) for future in futures]

    if get_meta:
        for i, (msg, tokens) in enumerate(results):

            if tokens is not None:
                price = compute_api_call_cost(
                    tokens['prompt_tokens'], tokens['completion_tokens'],
                    kwargs.get("model", "gpt-4-turbo-2024-04-09"))
            else:
                price = 0

            results[i][1] = price

    return results


def compute_api_call_cost(prompt_tokens: int,
                          completion_tokens: int,
                          model="gpt-4-turbo-2024-04-09"):
    """
    Warning: prices need to be manually updated from
    https://openai.com/api/pricing/
    """
    prices_per_million_input = {
        "gpt-4o": 5,
        "gpt-4-turbo": 10,
        "gpt-4": 30,
        "gpt-3.5-turbo": 0.5
    }
    prices_per_million_output = {
        "gpt-4o": 15,
        "gpt-4-turbo": 30,
        "gpt-4": 60,
        "gpt-3.5-turbo": 1.5
    }
    if "gpt-4o" in model:
        key = "gpt-4o"
    elif "gpt-4-turbo" in model:
        key = "gpt-4-turbo"
    elif 'gpt-4' in model:
        key = "gpt-4"
    elif 'gpt-3.5-turbo' in model:
        key = "gpt-3.5-turbo"

    price = prompt_tokens * prices_per_million_input[
        key] + completion_tokens * prices_per_million_output[key]
    price = price / 1e6

    return price


# basic testing
if __name__ == "__main__":
    import time
    import sys
    sys.path.insert(0, "..")
    sys.path.insert(0, ".")

    text0 = "How did Steve Irwin die?"
    # text1 = "what is your favourite australian animal?"
    # text2 = "how many chucks could a wood chuck chuck?"
    # text3 = "how fast does a lemur move?"
    # text4 = "who has the best llm?"

    model = "gpt-3.5-turbo"
    # # model = "gpt-4-turbo-2024-04-09"
    msg, res = call_gpt(text0, cache=False, json_mode=False)
    # res = call_gpt_batch([text0, text0], cache=False, json_mode=False)
    ipdb.set_trace()
    pass

    # texts = [text0, text1, text2, text0, text1]

    # if 1:
    #     # run async
    #     start = time.time()
    #     results0 = call_gpt_batch(texts=texts,
    #                               cache=False,
    #                               json_mode=False,
    #                               model=model)
    #     print(time.time() - start, f"parallel for {len(results0)} results")

    # # run sync
    # start = time.time()
    # for text in texts:
    #     res, msg = call_gpt(text=text,
    #                         cache=False,
    #                         json_mode=False,
    #                         model=model)
    # print(time.time() - start, f"sync for {len(texts )} results")
    # ipdb.set_trace()
    # pass