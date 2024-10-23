#!/usr/bin/env python3
"""dataset_wrappers.py in src/microchat/custom_datasets.

Adapted from:
https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/datasets/dataset_wrappers.py
"""
# sourcery skip: upper-camel-case-classes, no-loop-in-tests, no-conditionals-in-tests
__all__ = [
    "HotPotQAWrapper",
    "SciEvalWrapper",
]

from typing import Optional


from dotenv import find_dotenv
from dotenv import load_dotenv


import dspy
from dspy.datasets import HotPotQA
from loguru import logger

from microchat.custom_datasets.base_dataset import HFDataset

load_dotenv(find_dotenv())
RANDOM_SEED = 8675309


class HotPotQAWrapper:
    @staticmethod
    def __call__(
        # dataset_name: str,
        root: str = None,
        split: str = "train",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a HotPotQA dataset object."""

        return HotPotQA(
            train_seed=random_seed,
            train_size=20,
            eval_seed=random_seed + 1,
            dev_size=50,
            test_size=0,
        )


# OpenDFM / SciEval
class SciEvalWrapper:

    @staticmethod
    def __call__(
        dataset_name: str = "OpenDFM/SciEval",
        split: str = "validation",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a SciEval dataset object."""

        return HFDataset(
            dataset_name=dataset_name,
            split=split,
            train_seed=random_seed,
            # train_size=20,
            # dev_seed=random_seed + 1,
            # dev_size=50,
            # test_size=0,
            **kwargs,
        )
