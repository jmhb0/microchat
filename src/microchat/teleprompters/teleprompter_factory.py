#!/usr/bin/env python3
"""teleprompter_factor.py in src/microchat/teleprompters."""
from microchat.models.model_factory import create_model
from microchat.teleprompters.teleprompter_registry import OptimizerType

from typing import Tuple, Optional
import re

from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger

from microchat import MODULE_ROOT
from microchat.fileio.text.readers import yaml_loader
from microchat.models.base_llmodel import LLModel
from microchat.models.model_registry import ModelType
import dspy
import tiktoken as tk
import pprint as pp

# set ENV vars from .env
load_dotenv(find_dotenv())


def create_optimizer(
        optimizer_name: str,
        metric: Optional[dspy.Evaluate] = None,
        teacher_model: Optional[str] = "o1-mini",
        config: Optional[dict] = None,
        kwargs: Optional[dict] = None
) -> dspy.teleprompt.Teleprompter:
    """Create an optimizer model name.

    Args:
        optimizer_name (str): The type of optimizer to create, specified as an enum value.
        metric (Optional[dspy.Evaluate]): The metric to use for optimization. (default: "bootstrap_few_shot_random")


    Returns:
        dspy.teleprompt.Teleprompter: An optimizer model
    """
    # convert model_type to ModelType enum
    optimizer_type = OptimizerType[optimizer_name]
    if not isinstance(optimizer_type, OptimizerType):
        raise ValueError(f"Optmizer {optimizer_name} not found in OptimizerType enum.")

    # set metric if not provided
    if not metric:
        metric = dspy.evaluate.answer_exact_match
        logger.info(f"Using default metric: {metric.__name__}")
    elif not isinstance(metric, dspy.Evaluate):
        logger.error(f"Metric {metric} is not an instance of dspy.Evaluate.")
        raise ValueError(f"Metric {metric} is not an instance of dspy.Evaluate.")

    # update config
    config = config or {}
    config.update(kwargs or {})
    default_config = yaml_loader(MODULE_ROOT.joinpath("conf", "opt_config.yaml"))
    default_config = default_config.get(optimizer_type.name, {})

    # load model
    logger.info(f"Loading optimizer: {optimizer_type.name}")
    match optimizer_type:
        case OptimizerType.bootstrap_optuna:
            optimizer = dspy.BootstrapFewShot(metric=metric, **config)

        case OptimizerType.bootstrap_random:
            teacher_model = create_model(teacher_model)
            optimizer = dspy.BootstrapFewShotWithRandomSearch(
                metric=metric,
                teacher_settings=dict(lm=teacher_model.lm),
                **config
            )
        case OptimizerType.bootstrap_optuna:
            config = {

            }

        case _:  # no match
            raise NotImplementedError(
                f"Optimizer {optimizer_name} not implemented."
            )

    # update class attr
    optimizer.model_name = optimizer_type.value[-1]
    optimizer.config = dict(metric=metric, **config)

    return optimizer