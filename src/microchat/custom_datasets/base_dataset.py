#!/usr/bin/env python3
"""base_dataset in src/microchat/custom_datasets."""
from typing import List
from typing import Optional, Dict, Any

from loguru import logger


import random

from datasets import load_dataset

from dspy.datasets.dataset import Dataset
import dspy

from tqdm import tqdm


class HFDataset(Dataset):
    dataset_name: str

    def __init__(
        self, *args, dataset_name: str, split: str = "test", **kwargs: Optional[dict]
    ):
        # initialize the base class
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.kwargs: dict = kwargs

        # read splits
        hf_dataset: dict = {}
        # find if split in dataset
        available_splits = list(
            load_dataset(self.dataset_name, trust_remote_code=True).keys()
        )
        for split in available_splits:
            try:
                hf_dataset[split] = self.convert_hf_to_dspy(
                    dataset_name=self.dataset_name, split=split
                )
            except Exception as e:
                logger.error(f"Error loading dataset {self.dataset_name} split {split}")
                logger.error(e)

        # create unofficial dev and validation if not present
        if (
            hf_dataset.get("train")
            and not hf_dataset.get("validation")
            and not hf_dataset.get("dev")
        ):
            logger.info("Creating unofficial dev splits.")
            official_train = hf_dataset.pop("train")
            rng = random.Random(0)
            rng.shuffle(official_train)
            hf_dataset["train"] = official_train[: len(official_train) * 75 // 100]
            hf_dataset["dev"] = official_train[len(official_train) * 75 // 100 :]
        elif (
            not hf_dataset.get("train")
            and hf_dataset.get("validation")
            or hf_dataset.get("dev")
        ):
            logger.info("Creating unofficial train split from official dev.")
            official_dev = hf_dataset.pop("validation") or hf_dataset.pop("dev")
            rng = random.Random(0)
            rng.shuffle(official_dev)
            hf_dataset["train"] = official_dev[: len(official_dev) * 75 // 100]
            hf_dataset["dev"] = official_dev[len(official_dev) * 75 // 100 :]

        # assign splits
        self._train = hf_dataset.get("train", [])
        self._dev = hf_dataset.get("validation", []) or hf_dataset.get(
            "dev", []
        )  # dspy.Dataset uses "_dev" for validation
        self._test = hf_dataset.get("test", [])

    @staticmethod
    def convert_hf_to_dspy(
        dataset_name: str,
        split: str,
        question_key: str = "question",
        answer_key: str = "answer",
        keep_details: bool = True,
        trust_remote_code: bool = True,
    ) -> List[Dict[str, Any]]:
        """Convert a HuggingFace dataset to a DSPy dataset."""
        # load dataset
        hf_dataset = load_dataset(
            path=dataset_name, split=split, trust_remote_code=trust_remote_code
        )
        # check keys "question" and "answer" are present
        if (
            question_key not in hf_dataset.features
            or answer_key not in hf_dataset.features
        ):
            raise ValueError(
                f"Dataset {dataset_name} does not have 'question' and 'answer' fields."
            )

        # initialize the dspy dataset
        dataset: List[dspy.Example] = []
        keys: List[str] = [question_key, answer_key]

        # iterate over HuggingFace dataset and convert to DSPy dataset
        for raw_example in tqdm(
            hf_dataset, desc=f"Converting {dataset_name} to DSPy format"
        ):
            # convert example to DSPy format
            if keep_details:
                # extend keys with additional fields, keep set of keys unique
                keys = list(set(keys + list(raw_example.keys())))

            example = {k: raw_example[k] for k in keys}
            dataset.append(example)

        return dataset
