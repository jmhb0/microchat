#!/usr/bin/env python3
"""mcq_metric.py in src/microchat/metrics."""

import dspy
from loguru import logger

from microchat import MODULE_ROOT
from microchat.fileio.text.readers import yaml_loader
from microchat.mc_questions.mcq import MCQ

from microchat.models.dspy_modules import CoTSelfCorrectRAG
from microchat.models.model_factory import create_model
from microchat.utils.process_text import process_blooms, compute_tokens

blooms_taxonomy: dict = yaml_loader(MODULE_ROOT.joinpath("conf", "blooms.yaml")).get(
    "taxonomy"
)

DEFAULT_TEACHER = create_model("o1-mini")
DEFAULT_BLOOMS_MODULE = CoTSelfCorrectRAG(context="blooms")


# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM


def validate_blooms(example, pred, trace=None):
    """Measure the accuracy of a predicted blooms taxonomy level"""
    if answer_EM := dspy.evaluate.answer_exact_match(example, pred):
        return float(answer_EM)

    # process answer
    gt_level, gt_name = process_blooms(example.answer)
    pred_level, pred_name = process_blooms(pred.answer)

    # weighted score based on level, name, and self-assessment match
    weights = {
        "level": 0.75,
        "level_diff": 0.25,
    }
    level_score = 1 if gt_level == pred_level else 0
    # penalize larger differences
    level_diff = abs(gt_level - pred_level) ** 4
    level_diff = 1 / (1 + level_diff)

    # calculate weighted score
    score = zip([level_score, level_diff], weights.values())
    return sum(s * w for s, w in score)


def validate_nbme(example, pred, trace=None):
    if answer_EM := dspy.evaluate.answer_exact_match(example, pred):
        return float(answer_EM)

    # mcq model
    try:
        mcq = MCQ(example=example, prediction=pred)
    except Exception as e:
        logger.error(f"Error creating MCQ: {e}")
        return 0

    # weighted score based on level, name, and self-assessment match
    weights = {
        "similarity": 1.0,  # similarity between predicted and ground truth answers
        "formatted": 0.24,  # formatted according to NBME guidelines
        "extraneous": 0.5,  # extraneous information to give away the answer lower score
        "option_token_ratio": 0.5,  # if MC options, ratio of mean(incorrect) to correct
        "answer_token_metric": 0.5,  # exponential decay function for answer token difference
        "errors": 0.25,  # errors in parsing the example or prediction
    }
    # normalize weights to sum to 1
    weights = {k: v / sum(weights.values()) for k, v in weights.items()}

    # calculate weighted score
    metrics_dict = mcq.metrics
    score = zip(
        [metrics_dict[k] for k in weights if k in metrics_dict], weights.values()
    )
    return sum(s * w for s, w in score)
