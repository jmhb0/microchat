#!/usr/bin/env python3
"""model_factory.py in src/microchat/models."""

from typing import Tuple
import re

from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger



from microchat.models.base_llmodel import LLModel
from microchat.models.model_registry import ModelType
import dspy
import tiktoken as tk

# set ENV vars from .env
load_dotenv(find_dotenv())


def create_model(    model_name: str) -> LLModel:
    """Create a model and its associated transformation given a model type and cache directory.

    Args:
        model_name (str): The type of model to create, specified as an enum value.

    Returns:
        Tuple[dspy.LM, tk.core.Encoding]: A tuple containing the model and its associated tokenizer.
    """
    # get model prefix (remove ending date or _mini)
    model_prefix = re.sub(r"\d{8}$|mini$|latest$|mini_\d{8}$", "", model_name)
    model_name = model_name.lower().replace("-", "_")

    # convert model_type to ModelType enum
    model_type = ModelType[model_name]
    if not isinstance(model_type, ModelType):
        raise ValueError(f"Model {model_name} not found in ModelType enum.")

    # load model
    logger.info(f"Loading model: {model_type.name}")
    # logger.info(f"Saving model to cache directory: {cache_dir}")
    match model_type:
        case ModelType.gpt_4o_mini:  # type: ignore[no-untyped-call]
            dspy_model = dspy.LM("/".join(model_type.value))
            dspy_model.model_name = model_type.value[-1]
            dspy_model.model_prefix = model_prefix

        case _: # no match
            raise NotImplementedError(f"Model {model_type} is not yet supported.")

    return LLModel(model = dspy_model)