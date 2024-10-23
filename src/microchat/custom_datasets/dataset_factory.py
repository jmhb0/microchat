#!/usr/bin/env python3
"""dataset_factory.py in src/microchat/custom_datasets.

Adapted from:
https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/datasets/factory.py
"""

import dspy
from microchat.custom_datasets.dataset_registry import DatasetType


def create_dataset(dataset_name: str) -> dspy.datasets.Dataset:
    """Create a dataset given its corresponding DatasetType enum value.

    Args:
        dataset_name (str): An enum value representing the dataset to be created.

    Returns:
        dspy.datasets.Dataset: The dataset object.

    """
    dataset_type = DatasetType[dataset_name]
    return dataset_type.value()
