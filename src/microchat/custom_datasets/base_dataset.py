#!/usr/bin/env python3
"""base_dataset in src/microchat/custom_datasets."""
import os
from typing import List
from typing import Optional, Dict, Any, Union
from pathlib import Path

import pandas as pd
from loguru import logger


import random

from datasets import load_dataset

from dspy.datasets.dataset import Dataset
import dspy

from tqdm import tqdm

from microchat.fileio.dataframe.readers import df_loader


class HFDataset(Dataset):
    dataset_name: str

    def __init__(self, *args, dataset_name: str, **kwargs: Optional[dict]):
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
        dataset: List[Dict[str, Any]] = []
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


class CSVDataset(Dataset):
    filepath: Union[str, Path, os.PathLike]

    def __init__(
        self,
        filepath: str,
        subset: Optional[list] = None,
        question_key: Optional[str] = "question",
        answer_key: Optional[str] = "answer",
        **kwargs: Optional[dict],
    ):
        # initialize the base class
        super().__init__(**kwargs)
        self.filepath = filepath
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File {filepath} not found.")

        self.dataset_name = Path(filepath).stem
        self.kwargs: dict = kwargs
        # question_key = kwargs.get("question_key", "question")
        # answer_key = kwargs.get("answer_key", "answer")

        # read data
        df = df_loader(filepath)

        # HACK create new column revised question-answer
        # strip ending newline
        df["description"] = (
            df["question"].copy().apply(lambda x: x.split(r"Question:")[0].strip())
        )
        df["original_answer"] = "Answer:\n" + df["question_and_answer"].copy().apply(
            lambda x: x.split(r"Answer:")[1].strip()
        )
        df["original_question_answer"] = (
            df["question"] + "\n\nAnswer:\n```" + df["answer_correct"] + "```"
        )
        df["revised_question_answer"] = (
            "Question:\n```"
            + df["revised_question"]
            + "```\n\nAnswer:\n```"
            + df["answer_correct"]
            + "```"
        )
        df["revised_question_answer_mc"] = (
            "Question:\n```"
            + df["revised_question"]
            + "\n\nAnswer:\n```"
            + df["answer_correct"]
            + "\n\nOptions:\n```"
            + df["multiple_choice"]
        )

        # strip ending newline or whitespace
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # assert question_key, answer_key
        col_set_diff = set([question_key, answer_key]) - set(df.columns)
        if col_set_diff:
            raise ValueError(
                f"DataFrame {filepath} does not have columns {col_set_diff}."
            )

        # create splits based on "split" column
        if "split" not in df.columns:
            df["split"] = "train"

        # create output dict
        df_dataset: dict = {}

        # find if split in dataset
        available_splits = df["split"].unique()
        for split in available_splits:
            try:
                df_dataset[split] = self.convert_df_to_dspy(
                    df=df,
                    subset=subset,
                    question_key=question_key,
                    answer_key=answer_key,
                )
            except Exception as e:
                logger.error(f"Error loading dataset {self.dataset_name} split {split}")
                logger.error(e)

        # create unofficial dev and validation if not present
        if (
            df_dataset.get("train")
            and not df_dataset.get("validation")
            and not df_dataset.get("dev")
        ):
            logger.info("Creating unofficial dev splits.")
            official_train = df_dataset.pop("train")
            rng = random.Random(0)
            rng.shuffle(official_train)
            df_dataset["train"] = official_train[: len(official_train) * 75 // 100]
            df_dataset["dev"] = official_train[len(official_train) * 75 // 100 :]
        elif (
            not df_dataset.get("train")
            and df_dataset.get("validation")
            or df_dataset.get("dev")
        ):
            logger.info("Creating unofficial train split from official dev.")
            official_dev = df_dataset.pop("validation") or df_dataset.pop("dev")
            rng = random.Random(0)
            rng.shuffle(official_dev)
            df_dataset["train"] = official_dev[: len(official_dev) * 75 // 100]
            df_dataset["dev"] = official_dev[len(official_dev) * 75 // 100 :]

        # assign splits
        self._train = df_dataset.get("train", [])
        self._dev = df_dataset.get("validation", []) or df_dataset.get(
            "dev", []
        )  # dspy.Dataset uses "_dev" for validation
        self._test = df_dataset.get("test", [])

    @staticmethod
    def convert_df_to_dspy(
        df: pd.DataFrame,
        question_key: str = "question",
        answer_key: str = "answer",
        keep_details: bool = False,
        subset: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert a HuggingFace dataset to a DSPy dataset."""
        # check keys "question" and "answer" are present
        if question_key not in df.columns or answer_key not in df.columns:
            raise ValueError("Dataset does not have 'question' and 'answer' fields.")

        # filter to remove rows with missing values
        if subset:
            # get num missing in subset
            num_missing = df.loc[:, subset].isnull().sum(axis=1)
            df = df.dropna(subset=subset)
            logger.info(f"Removed {num_missing.sum()} rows with missing values.")
            logger.info(f"Remaining rows: {len(df)}")

        # initialize the dspy dataset
        dataset: List[Dict[str, Any]] = []
        keys: List[str] = [
            question_key,
            answer_key,
            "key_image",
            "key_question",
            "blooms_reasoning",
        ]

        # iterate over HuggingFace dataset and convert to DSPy dataset
        for idx, raw_example in tqdm(
            df.iterrows(), desc="Converting df to DSPy format"
        ):
            # convert example to DSPy format
            if keep_details:
                # extend keys with additional fields, keep set of keys unique
                keys = list(set(keys + list(raw_example.keys())))

            # convert pandas Series to dict
            raw_example = raw_example.to_dict()
            example = {}
            for k in keys:
                if k == question_key:
                    # remove key from example and assign to question
                    question = raw_example.pop(k)
                    example["question"] = question.strip()
                elif k == answer_key:
                    answer = raw_example.pop(k)
                    example["answer"] = answer.strip()
                elif k in {"key_image", "key_question", "blooms_reasoning"}:
                    example[k] = raw_example[k]
                else:
                    continue
                    # example[k] = example[k].strip()

            dataset.append(example)

        return dataset
