#!/usr/bin/env python3
"""convert_hf_to_csv.py in src/microchat."""
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger

from microchat import MODULE_ROOT, DATA_ROOT

import datasets # HF datasets


def extract_questions(row):
    """Extract questions from row for microbench dataset.

    Will need to be customized for other HF datasets"""

    all_question = []
    for q_type in row["questions"].keys():
        elem = row["questions"][q_type]
        if elem is None:
            logger.warning(f"Skipping {q_type} for {row}")
            continue

        question_id = elem["id"] # 'question_id' is col 'id' for ubench (UUID)
        question_stem = elem["question"] # 'question_stem' is col 'question' for ubench
        correct_answer = elem["answer"] # 'correct_answer' is col 'answer' for ubench
        multiple_choice = elem["options"] # MC col is "options" for ubench
        all_question.append(
            {
                "source": row["source"], # dataset
                "chapter": row["chapter"], # dummy (not needed for HF datasets, used for textbook questions)
                "question_id": question_id,
                "question_stem": question_stem,
                "correct_answer": correct_answer,
                "multiple_choice": multiple_choice,
                "question_type": q_type,
            }
        )

    return all_question


@click.command()
@click.option("--dataset", type=click.STRING, default="jnirschl/uBench")
@click.option(
    "--output-dir", type=click.Path(file_okay=False, exists=False, path_type=Path)
)
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    dataset: str,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    """Extract questions from HF dataset and save to CSV."""
    output_dir = output_dir or Path(DATA_ROOT).joinpath("dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        MODULE_ROOT.joinpath(f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Dataset: {dataset}")
    logger.info(f"Output directory: {output_dir}")

    # load hf dataset
    hf_dataset = datasets.load_dataset(dataset, split="test")
    logger.info(f"Dataset loaded: {hf_dataset}")

    # loop over dataset and save question_stem, correct_answer to CSV
    # group by col datatset
    unique_classes = hf_dataset.unique("label_name")
    logger.info(f"Unique classes: {unique_classes}")


    # convert to df
    df = hf_dataset.to_pandas()
    df["source"] = dataset.split("/")[-1] # dataset name
    df["chapter"] = None # dummy (not needed for HF datasets, used for textbook questions)

    # get first example for each class (only need one label per class for training)
    first_example = df.groupby("label_name").first()
    logger.info(f"First example: {first_example}")

    #
    if dataset.split("/")[-1].lower() != "uBench":
        logger.warning(f"Please customize 'extract_questions' function for dataset: {dataset}")

    # for each example, save question_stem, correct_answer to CSV
    output_list = []
    for idx, row in first_example.iterrows():
        output_list.extend(extract_questions(row))

    # combine all data
    output_df = pd.DataFrame(output_list)
    logger.info(f"Output df: {output_df}")
    if dry_run:
        logger.info("Dry run: no changes will be made.")
    else:
        output_file = output_dir.joinpath("questions.csv")
        output_df.to_csv(output_file, index=False)
        logger.info(f"Questions saved to {output_file}")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
