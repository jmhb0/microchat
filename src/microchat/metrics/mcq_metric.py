#!/usr/bin/env python3
"""mcq_metric.py in src/microchat/metrics."""

import dspy

from microchat import MODULE_ROOT
from microchat.fileio.text.readers import yaml_loader

from microchat.models.dspy_modules import CoTSelfCorrectRAG
from microchat.models.model_factory import create_model
from microchat.utils.process_text import process_blooms

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
    if answer_EM := dspy.evaluate.answer_exact_match(example, pred):
        return int(answer_EM)

    # process answer
    gt_level, gt_name = process_blooms(example.answer)
    pred_level, pred_name = process_blooms(pred.answer)

    # weighted score based on level, name, and self-assessment match
    weights = {
        "level": 0.75,
        "name": 0.25,
    }
    # check if the predicted answer is in blooms taxonomy
    level_score = 1 if gt_level == pred_level else 0
    name_score = 1 if gt_name == pred_name else 0

    # calculate weighted score
    score = sum([level_score, name_score])
    # if trace is not None:
    #     trace.update({"level_score": level_score, "name_score": name_score})

    return sum(score * weight for weight in weights.values())