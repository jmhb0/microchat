#!/usr/bin/env python3
"""run_dspy.py in src/microchat."""
from pathlib import Path
from typing import Optional
import click
import pandas as pd
from dotenv import find_dotenv
from dotenv import load_dotenv

from loguru import logger

import dspy
from dspy.evaluate.evaluate import Evaluate
from tqdm import tqdm

from microchat import PROJECT_ROOT
from microchat.custom_datasets.dataset_factory import create_dataset
from microchat.fileio.dataframe.readers import df_loader
from microchat.models.dspy_modules import CoTSelfCorrectRAG
from microchat.models.model_factory import create_model
from microchat.mc_questions.mcq import MCQ, Blooms
from microchat.teleprompters.teleprompter_factory import create_optimizer

try:
    import datasets

    if datasets.__version__ != "3.0.1":
        raise ImportError(
            f"Dataset may not be compatible with DSPy. Please install datasets==3.0.1."
        )
except ImportError as e:
    logger.error("Please install datasets==3.0.1.")
    logger.error(e)
    raise e


@click.command()
@click.argument("dataset_name", type=click.STRING)
@click.option(
    "--model", type=click.STRING, default="o1-mini"
)  # "gpt-4o-mini") # gpt-4o
@click.option("--teacher-model", type=click.STRING, default="gpt-4o")
@click.option("--retrieval-model", type=click.STRING, default="wiki17_abstracts")
@click.option("--optimizer", type=click.STRING, default="bootstrap_random")
@click.option(
    "--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option("--task", type=click.Choice(["nbme", "blooms"]), default="blooms")
@click.option("--random-seed", type=click.INT, default=8675309)
@click.option("--retrieve-k", type=click.IntRange(3, 10), default=5)
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    dataset_name: str,
    model: Optional[str] = "gpt-4o-mini",
    teacher_model: Optional[str] = "o1-mini",
    retrieval_model: Optional[str] = "wiki17_abstracts",
    optimizer: Optional[str] = "bootstrap",
    output_dir: Optional[Path] = None,
    task: Optional[str] = "blooms",
    random_seed: int = 8675309,
    retrieve_k: int = 5,
    dry_run: bool = False,
) -> None:
    """Docstring."""
    if not output_dir:
        output_dir = Path(PROJECT_ROOT).joinpath("outputs", dataset_name.strip(".csv"))
    output_dir.mkdir(parents=True, exist_ok=True)

    project_dir = Path(__file__).parents[2]
    logger.add(
        project_dir.joinpath("logs", f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model: {model}")
    logger.info(f"Teacher model: {teacher_model}") if teacher_model else None
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Output directory: {output_dir}")

    if "mini" not in model:
        logger.warning(f"Model {model} may require increased costs.")
        click.confirm(
            f"Are you sure you want to continue with {model} and increased costs?",
            abort=True,
        )

    if "mini" not in teacher_model:
        logger.warning(f"Teacher model {teacher_model} may require increased costs.")
        click.confirm(
            f"Are you sure you want to continue with {teacher_model} and increased costs?",
            abort=True,
        )

    if dry_run:
        logger.info("Dry run: no changes will be made.")
        return

    # instantiate model LLM/VLM model
    model = create_model(model)

    # define retrieval model
    colbertv2_model = None
    # if retrieval_model:
    #     logger.info(f"Retrieval model: {retrieval_model}")
    #     logger.info(f"Retrieve k: {retrieve_k}")
    #     colbertv2_model = dspy.ColBERTv2(
    #         url="http://20.102.90.50:2017/wiki17_abstracts"
    #     )

    # configure DSPy settings
    dspy.settings.configure(lm=model.lm, rm=colbertv2_model)

    # set task with question_key and answer_key
    # TODO: load config from yaml
    if task == "blooms":
        question_key = (
            "revised_question_answer"  # "question" #"original_question_answer"
        )
        answer_key = (
            "blooms_question_category"  # "revised_question" #"revised_question_answer"
        )
    else:
        logger.error(f"Task {task} not implemented.")
        raise NotImplementedError(f"Task {task} not implemented.")

    # instantiate dataset
    subset = [question_key, answer_key]
    dataset = create_dataset(
        dataset_name, subset=subset, question_key=question_key, answer_key=answer_key
    )

    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]

    print(f"{len(trainset)}, {len(devset)}")

    train_example = trainset[0]
    dev_example = devset[0]
    logger.info(f"Train question: {train_example.question}")
    logger.info(f"Train answer: {train_example.answer}")

    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

    # get rows with non null and non empty list
    idx_temp = temp.loc[
        temp["blooms_question_category_pred"].notnull()
        & temp["blooms_question_category_pred"].apply(lambda x: len(x) > 0)
    ].index
    temp = temp.loc[idx_temp]

    #
    temp.to_csv(
        output_dir.joinpath(f"{model.model_name}_update_blooms_agg.csv"), index=False
    )

    #
    module.save(output_dir.joinpath(f"{model.model_name}_demos.json"))

    # compile rag
    compiled_rag = optimizer.compile(module, trainset=trainset)

    # instantiate MCQ to test the compiled rag
    model_dump = model.model_dump(include={"tokenizer"})
    if task == "blooms":
        test_single = Blooms(example=dev_example, module=compiled_rag, **model_dump)
    elif task == "nbme":
        test_single = MCQ(
            example=dev_example,
            module=compiled_rag,
            **model.model_dump(include={"tokenizer"}),
        )
    else:
        logger.error(f"Task {task} not implemented.")

    ##
    # Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
    evaluator = Evaluate(
        devset=devset,
        num_threads=1,
        display_progress=True,
        display_table=5,
        provide_traceback=True,
    )

    # Evaluate the `compiled_rag` program with the `answer_exact_match` metric.
    metric = dspy.evaluate.answer_exact_match
    temp_eval = evaluator(compiled_rag, metric=metric)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
