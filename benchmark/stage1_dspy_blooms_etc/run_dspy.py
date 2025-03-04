#!/usr/bin/env python3
"""run_dspy.py in src/microchat."""
import pprint
from pathlib import Path
from typing import Optional
import click
import pandas as pd

from dotenv import find_dotenv
from dotenv import load_dotenv

from loguru import logger
from tqdm import tqdm

import dspy
from dspy.evaluate.evaluate import Evaluate

from microchat import PROJECT_ROOT
from microchat.custom_datasets.dataset_factory import create_dataset
from microchat.fileio.dataframe.readers import df_loader
from microchat.fileio.text.readers import yaml_loader
from microchat.fileio.text.writers import yaml_writer
from microchat.metrics.mcq_metric import (
    validate_blooms,
    validate_nbme,
    validate_tagging,
)
from microchat.models.dspy_modules import CoTSelfCorrectRAG, CoTRAG, context
from microchat.models.model_factory import create_model
from microchat.mc_questions.mcq import MCQ, Blooms
from microchat.teleprompters.teleprompter_factory import create_optimizer
from microchat.utils.process_model_history import history_to_jsonl

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

try:
    from langtrace_python_sdk import langtrace

    # langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))
except ImportError as e:
    logger.warning("Langtrace not installed.")

# import openai
# import random
# from random import shuffle

@click.command()
@click.argument("dataset_name", type=click.STRING)  # blooms.csv
@click.option("--model", type=click.STRING, default="o1-mini")
@click.option("--teacher-model", type=click.STRING, default="o1-mini")
@click.option("--retrieval-model", type=click.STRING, default=None)
@click.option("--optimizer", type=click.STRING, default="miprov2")
@click.option(
    "--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--task",
    type=click.Choice(["nbme", "blooms", "hotpotqa", "organism_research"]),
    default="blooms",
)
@click.option("--random-seed", type=click.INT, default=8675309)
@click.option("--retrieve-k", type=click.IntRange(1, 10), default=5)
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
    logger.info(f"Task: {task}")
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
    teacher_lm = None  # create_model(teacher_model)

    # define retrieval model
    colbertv2_model = None
    if retrieval_model or task == "hotpotqa":
        logger.info(f"Retrieval model: {retrieval_model}")
        logger.info(f"Retrieve k: {retrieve_k}")
        colbertv2_model = dspy.ColBERTv2(
            url="http://20.102.90.50:2017/wiki17_abstracts"
        )

    # configure DSPy settings
    dspy.settings.configure(lm=model.lm, rm=colbertv2_model)

    ### parse MCQ options

    # temp
    # input_file = Path("/home/jjn/GitHub/microchat/data/processed/language_bias").joinpath("language_bias.csv")
    input_file = Path("/home/jjn/GitHub/microchat/data/processed/").joinpath(
        "llm-qc_eval_naive.csv"
    )
    df = df_loader(input_file)

    df["self_assess_llm"] = df["self_assess_llm"].fillna("")

    # apply MCQ to col "question_answer" in df
    # create MCQ object
    start_idx = -1
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx < start_idx:
            continue

        # strip ending newline
        text = row["naive_question_answer_formatted"]

        if row["self_assess_llm"] and not pd.isna(row["similarity_metric"]):
            logger.info(f"Skipping index {idx}, which was already reviewed")
            continue

        # format example and prediction
        example = dspy.Example(
            question=row["description_question_answer"],
            key_question=row["key_question"],
            key_image=row["key_image"],
        )
        pred = dspy.Prediction(answer=text)
        try:
            mcq = MCQ(example=example, prediction=pred)

            # info
            if df.loc[idx, "description"] != mcq.example_dict.get("description"):
                logger.warning(f"Description mismatch at index {idx}")

            if df.loc[idx, "original_question"] != mcq.example_dict.get("question"):
                logger.warning(f"Original question mismatch at index {idx}")

            if (
                df.loc[idx, "original_answer"]
                != mcq.example_dict.get("correct_answer").strip()
            ):
                logger.warning(f"Original answer mismatch at index {idx}")

            # update df
            df.loc[idx, "similarity_metric"] = mcq.metrics.get("similarity")
            df.loc[idx, "formatted_metric"] = mcq.metrics.get("formatted")
            df.loc[idx, "extraneous_metric"] = mcq.metrics.get("extraneous")
            df.loc[idx, "option_token_ratio_metric"] = mcq.metrics.get(
                "option_token_ratio"
            )
            df.loc[idx, "reasoning"] = mcq.metrics.get("reasoning", None)
            df.loc[idx, "self_assess_llm"] = model.model_name

            df.loc[idx, "description"] = mcq.example_dict.get("description")
            df.loc[idx, "additional_info"] = mcq.example_dict.get("additional_info")
            df.loc[idx, "original_question"] = mcq.example_dict.get("question")
            df.loc[idx, "original_question_tokens"] = mcq.get_tokens(
                original=True, field="question"
            )
            df.loc[idx, "original_answer"] = mcq.example_dict.get(
                "correct_answer"
            ).strip()
            df.loc[idx, "original_answer_tokens"] = mcq.get_tokens(
                original=True, field="correct_answer"
            )
            df.loc[idx, "revised_question"] = mcq.prediction_dict.get("question")
            df.loc[idx, "revised_question_tokens"] = mcq.get_tokens(
                original=False, field="question"
            )
            df.loc[idx, "revised_answer"] = mcq.prediction_dict.get(
                "correct_answer"
            ).strip()
            df.loc[idx, "revised_answer_tokens"] = mcq.get_tokens(
                original=False, field="correct_answer"
            )
            df.loc[idx, "options"] = pprint.pformat(mcq.prediction_dict["options"])
            df.loc[idx, "correct_index"] = mcq.prediction_dict["correct_index"]
            df.to_csv(input_file, index=False)
        except Exception as e:
            logger.error(f"Error creating MCQ for index {idx}: {e}")
            df.to_csv(input_file, index=False)

    # # save
    df.to_csv(input_file, index=False)

    # set task with question_key and answer_key
    # TODO: load config from yaml
    if task == "blooms":
        question_key = "question_answer"  # "revised_question_answer"  # "question" #"original_question_answer"
        answer_key = "blooms_question_category"  # "blooms_question_category"  # "revised_question" #"revised_question_answer"
        metric = validate_blooms
        eval_metric = dspy.evaluate.answer_exact_match
        subset = [question_key, answer_key]
    elif task == "nbme":
        question_key = "question"  # "original_question_answer"
        answer_key = "answer"  # "revised_question_answer"
        metric = validate_nbme
        eval_metric = validate_nbme
        subset = [question_key, answer_key]
    elif task == "hotpotqa":
        question_key = "question"
        answer_key = "answer"
        metric = dspy.evaluate.answer_exact_match
        eval_metric = dspy.evaluate.answer_exact_match
        subset = [question_key, answer_key]
    elif task == "organism_research":
        question_key = "description_question_answer"
        answer_key = "original_answer"
        metric = validate_tagging
        eval_metric = validate_tagging
        subset = [question_key, answer_key, "organism", "specimen", "research_subject"]
    else:
        logger.error(f"Task {task} not implemented.")
        raise NotImplementedError(f"Task {task} not implemented.")

    # instantiate dataset
    dataset = create_dataset(
        dataset_name, subset=subset, question_key=question_key, answer_key=answer_key
    )

    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs("context", "question") for x in dataset.train]
    devset = [x.with_inputs("context", "question") for x in dataset.dev]
    if not trainset or not devset:
        logger.error(f"Empty dataset: {dataset_name}")
        raise ValueError(f"Empty dataset: {dataset_name}")

    print(f"{len(trainset)}, {len(devset)}")
    #
    # train_example = trainset[0]
    # dev_example = devset[0]
    # logger.debug(f"Train question: {train_example.question}")
    # logger.debug(f"Train answer: {train_example.answer}")

    # Set up a teleprompter/optimizer, which will compile our RAG program.
    optimizer, metric = create_optimizer(
        optimizer, teacher_model=teacher_model, metric=metric
    )

    # create module
    module = CoTRAG(context=task, num_passages=retrieve_k)
    module.name = module.__class__.__name__

    # Set up the `evaluator` function. We'll use this many times below.
    evaluator = Evaluate(
        devset=devset,
        num_threads=1,
        display_progress=True,
        provide_traceback=True,
        metric=eval_metric,
        return_outputs=True,
    )

    # # results yaml
    results_file = output_dir.joinpath("results.yaml")
    model_filepath = output_dir.joinpath(
        f"{model.model_name}_{module.name}_{module.signature_name}.json"
    )

    # evalute zero shot
    zs_score, zs_results = evaluator(module, metric=eval_metric)
    logger.info(f"Zero-shot score: {zs_score}")
    if results_file.exists():
        results = yaml_loader(results_file)
        if model.model_name not in results:
            results[model.model_name] = {}
        results[model.model_name][module.signature_name]["zero_shot_score"] = float(
            zs_score
        )
        yaml_writer(results, results_file)
    else:
        yaml_writer(
            {
                model.model_name: {
                    module.signature_name: {"zero_shot_score": float(zs_score)}
                }
            },
            results_file,
        )

    ##### Microchat processing
    # read saved df
    ##### Microchat processing
    # output_csv = output_dir.joinpath(f"{model_filepath.stem}_output_processed.csv")
    # output_csv = Path("/home/jjn/GitHub/microchat/data/processed").joinpath("o1-mini_CoTRAG_SelfAssessRevisedInput_output_processed_temp.csv")
    # postbot
    # output_csv = Path("/home/jjn/GitHub/microchat/data/processed").joinpath("o1-mini_CoTRAG_SelfAssessRevisedInput_output_processed_postbot.csv")
    # output_df = pd.read_csv(output_csv)

    # # scratch
    # output_csv = Path("/home/jjn/GitHub/microchat/data/processed").joinpath("blooms_classification_finetuning_v2.csv")
    # output_df = pd.read_csv(output_csv)
    #
    # # # save
    # # output_df.to_csv(output_csv, index=False)
    #
    # # update question_type, if nan fill with "blooms_level"
    # output_df["question_type_2"] = output_df.apply(lambda x: x["blooms_level"] if pd.isna(x["question_type"]) else x["question_type"], axis=1)
    #
    # # parse df into different groups based on "source"
    # new_df = []
    # for source, group in output_df.groupby("source"):
    #     logger.info(f"Processing {source} with {len(group)} examples")
    #     if len(group) < 200:
    #         new_df.append(group)
    #         continue
    #     # sample 20 per unique "question" type
    #     group = group.copy()
    #     group = group.reset_index(drop=True)
    #     # sample 20 per  "question_type"
    #     group_2 = group.copy().groupby("question_type_2")
    #     temp = group_2.apply(lambda x: x.sample(min(20, len(x))))
    #     # reset index
    #     temp = temp.reset_index(drop=True)
    #     new_df.append(temp)
    #
    # # concat new_df
    # output_df = pd.concat(new_df)
    #
    #
    #
    # #
    # def sample_group(group, sample_col: str="question_type", sample_size: int = 50):
    #     # if group size < 200, sample all
    #     if len(group) < 200:
    #         return group
    #
    #     # if col question_type in group and not empty/nan
    #     # sample min(20, len(group_2)) per group
    #     if sample_col in group.columns:
    #         group_2 = group.copy().groupby("question_type")
    #         temp = group_2.apply(lambda x: x.sample(min(sample_size, len(x))))
    #         # reset index
    #         temp = temp.reset_index(drop=True)
    #     else:
    #         # sample by "blooms_level"
    #         group_2 = group.copy().groupby("blooms_level")
    #         return group_2.apply(lambda x: x.sample(min(sample_size, len(x))))
    #
    #
    #
    #
    # # sample 20 per group
    # new_df = df_group.apply(sample_group)
    #
    # # if postbot
    # if "postbot" in output_csv.stem:
    #     # create new col with formatted revised question answer parsing mcq
    #     # convert json in "choices_postbot"
    #     # "{'choices': ['The precision of cooling processes determines whether proteins maintain their correct configuration.', 'Adjustments in electron flow intensity during imaging enhance the accuracy of capturing intricate details.', 'Variations in image processing software introduce significant differences in the crystallinity of images.', 'Changes in environmental humidity impact the hydration levels of proteins, leading to observable structural differences.', 'Gradual changes in radiation exposure create minute but noticeable diversions in structural clarity.'], 'correct_index': 0}"
    #     output_df["choices_postbot"] = output_df["choices_postbot"].apply(lambda x: eval(x))
    #     output_df["answer_postbot"] = output_df["choices_postbot"].apply(lambda x: x["choices"][x["correct_index"]])
    #     output_df["correct_index_postbot"] = output_df["choices_postbot"].apply(lambda x: x["correct_index"])
    #     # choices_formatted_postbot - format as A) B) C) D) E)
    #     output_df["choices_formatted_postbot"] = output_df["choices_postbot"].apply(
    #         lambda x: "\n".join([f"{chr(65 + i)}) {choice}" for i, choice in enumerate(x["choices"])])
    #     )
    #     output_df["answer_formatted_postbot"] = output_df["choices_postbot"].apply(
    #         lambda x: f"{chr(65 + x['correct_index'])}) {x['choices'][x['correct_index']]}"
    #     )
    #
    #
    #     # find any nan in "choices_postbot" or "answer_postbot"
    #     idx_nan = output_df[output_df["question_postbot"].isna() | output_df["choices_postbot"].isna() | output_df["answer_postbot"].isna() | output_df["choices_formatted_postbot"].isna()].index
    #     # drop rows with nan
    #     if len(idx_nan) > 0:
    #         logger.info(f"Found {len(idx_nan)} nan in choices_postbot or answer_postbot")
    #         output_df = output_df.drop(idx_nan)
    #
    #     output_df["revised_question_answer_postbot"] = "Question:\n```" + output_df["question_postbot"] + "\n\n" + output_df["choices_formatted_postbot"] +  "\n\nCorrect Answer: " + output_df["answer_formatted_postbot"] + "```"

    # # llm refined questions
    # llm_revised_df = Path("/home/jjn/GitHub/microchat/data/processed").joinpath("VQA_3_bot_refined - exp_1105_test150_dspy_150_best_5.csv.csv")
    # llm_revised_df = pd.read_csv(llm_revised_df)
    # logger.info(f"Processing {len(output_df)} examples")
    #
    # llm_revised_df_subset = llm_revised_df[["key_image", "key_question", 'log_str', 'iterations', 'use_case_r', 'code', 'cost', 'question_key', 'mcqs', 'mcqs_formatted', 'question_postbot', 'choices_postbot', 'question_answer_pass', 'mcq_semantic_pass', 'mcqs_format_pass', 'jeff_notes', 'jeff_code']]
    #
    # # inner join to only keep rows with key_image and key_question in both dfs
    # output_df = output_df.merge(llm_revised_df_subset, on=["key_image", "key_question"], how="inner")
    #
    # # save new name
    # output_df.to_csv(output_dir.joinpath(f"{model_filepath.stem}_output_processed_postbot.csv"), index=False)

    # # add empty output rows
    # output_df["similarity_metric"] = None
    # output_df["formatted_metric"] = None
    # output_df["extraneous_metric"] = None
    # output_df["option_token_ratio_metric"] = None
    # output_df["reasoning"] = None

    # output_df["option_token_ratio_metric"] = None
    # output_df["description"] = None
    # output_df["additional_info"] = None
    # output_df["original_question"] = None
    # output_df["original_question_tokens"] = None
    # output_df["original_answer"] = None
    # output_df["original_answer_tokens"] = None
    # output_df["revised_question"] = None
    # output_df["revised_question_tokens"] = None
    # output_df["revised_answer"] = None
    # output_df["revised_answer_tokens"] = None
    # output_df["options"] = None

    # # process col `revised_question_answer`
    # start_idx = -1
    # for idx, row in tqdm(output_df.iterrows(), total=len(output_df)):
    #     if idx < start_idx:
    #         continue
    #
    #     # strip ending newline
    #     text = row["revised_question_answer"]
    #     # revised_question_answer_postbot
    #     text = row["revised_question_answer_postbot"]
    #
    #     # if not pd.isna(row["similarity_metric"]) and not pd.isna(row["formatted_metric"]):
    #     #     continue
    #     if row["self_assess_llm"] == "o1-preview":
    #         logger.info(f"Skipping index {idx}, which was already reviewed")
    #         continue
    #
    #     # format example and prediction
    #     example = dspy.Example(
    #         question=row["description_question_answer"],
    #         key_question=row["key_question"],
    #         key_image=row["key_image"],
    #     )
    #     output = dspy.Prediction(answer=text)
    #     try:
    #         mcq = MCQ(example=example, prediction=output)
    #
    #         if output_df.loc[idx, "correct_index_postbot"] != mcq.prediction_dict["correct_index"]:
    #             logger.warning(f"Correct index mismatch at index {idx}")
    #
    #         # # update df
    #         output_df.loc[idx, "similarity_metric"] = mcq.metrics.get("similarity")
    #         output_df.loc[idx, "formatted_metric"] = mcq.metrics.get("formatted")
    #         output_df.loc[idx, "extraneous_metric"] = mcq.metrics.get("extraneous")
    #         output_df.loc[idx, "option_token_ratio_metric"] = mcq.metrics.get("option_token_ratio")
    #         output_df.loc[idx, "reasoning"] = mcq.metrics.get("reasoning", None)
    #         output_df.loc[idx, "self_assess_llm"] = model.model_name
    #
    #         # # update df
    #         # output_df.loc[idx, "nbme_formatted"] = mcq.metrics.get("nbme_formatted")
    #         # output_df.loc[idx, "question_flaws"] = mcq.metrics.get("question_flaws")
    #         # output_df.loc[idx, "answer_flaws"] = mcq.metrics.get("answer_flaws")
    #         # output_df.loc[idx, "distractor_flaws"] = mcq.metrics.get("distractor_flaws")
    #         # output_df.loc[idx, "flaws_reasoning"] = mcq.metrics.get("reasoning", None)
    #         # output_df.loc[idx, "self_assess_llm_2"] = model.model_name
    #
    #
    #
    #         # info
    #         output_df.loc[idx, "description"] = mcq.example_dict.get("description")
    #         output_df.loc[idx,"additional_info"] = mcq.example_dict.get("additional_info")
    #         output_df.loc[idx,"original_question"] = mcq.example_dict.get("question")
    #         output_df.loc[idx,"original_question_tokens"] = mcq.get_tokens(
    #             original=True, field="question"
    #         )
    #         output_df.loc[idx,"original_answer"] = mcq.example_dict.get("correct_answer").strip()
    #         output_df.loc[idx,"original_answer_tokens"] = mcq.get_tokens(
    #             original=True, field="correct_answer"
    #         )
    #         output_df.loc[idx,"revised_question"] = mcq.prediction_dict.get("question")
    #         output_df.loc[idx,"revised_question_tokens"] = mcq.get_tokens(
    #             original=False, field="question"
    #         )
    #         output_df.loc[idx,"revised_answer"] = mcq.prediction_dict.get("correct_answer").strip()
    #         output_df.loc[idx,"revised_answer_tokens"] = mcq.get_tokens(
    #             original=False, field="correct_answer"
    #         )
    #         output_df.loc[idx, "options"] = pprint.pformat(mcq.prediction_dict["options"])
    #         output_df.loc[idx, "correct_index"] = mcq.prediction_dict["correct_index"]
    #
    #         # if idx % 10 == 0:
    #         output_df.to_csv(output_csv, index=False)
    #         output_df.to_csv(output_dir.joinpath(f"{model_filepath.stem}_output_processed_temp.csv"), index=False)
    #
    #     except Exception as e:
    #         logger.error(f"Error creating MCQ for index {idx}: {e}")
    #         output_df.to_csv(output_dir.joinpath(f"{model_filepath.stem}_output_processed_temp.csv"), index=False)
    #
    # # save to csv
    # output_df.to_csv(output_dir.joinpath(f"{model_filepath.stem}_output_processed.csv"), index=False)
    #
    # return

    # ##### Microchat inference #####
    # if not model_filepath.exists():
    #     logger.error(f"Error saving compiled RAG to {model_filepath}")
    #     raise FileNotFoundError(f"Error saving compiled RAG to {model_filepath}")
    #
    # logger.info(f"Loading compiled RAG from {model_filepath}")
    # trained_module = CoTRAG(context=task, num_passages=retrieve_k)
    # trained_module.load(model_filepath)
    # trained_module.name = trained_module.__class__.__name__
    #
    # # read df
    # df = df_loader(Path(DATA_ROOT).joinpath(dataset_name))
    # logger.info(f"Processing {len(df)} examples")
    #
    # # strip ending newline
    # df["description"] = (
    #     df["question"].copy().apply(lambda x: x.split(r"Question:")[0].strip())
    # )
    # # df["original_answer"] = "Answer:\n" + df[
    # #     "question_and_answer"
    # # ].copy().apply(lambda x: x.split(r"Answer:")[1].strip())
    # df["original_question_answer"] = (
    #     df["question"] + "\n\nAnswer:\n```" + df["answer"] + "```"
    # )
    # logger.info(f"Example input:\t{df['original_question_answer'].iloc[0]}")
    # # df["revised_question_answer"] = (
    # #         "Revised Question:\n```"
    # #         + df["revised_question"]
    # #         + "```\n\nRevised Answer:\n```"
    # #         + df["answer_correct"]
    # #         + "```"
    # # )
    # # df["revised_question_answer_mc"] = (
    # #         "Revised Question:\n```"
    # #         + df["revised_question"]
    # #         + "\n\nRevised Answer:\n```"
    # #         + df["answer_correct"]
    # #         + "\n\nOptions:\n```"
    # #         + df["multiple_choice"]
    # # )
    #
    # # drop nan for col original_question_answer
    #
    # # create examples
    # nbme_context = context["nbme"]
    # nbme_formatted = trained_module._format_context(nbme_context)
    # examples = []
    # for idx, row in df.iterrows():
    #     # if row["key_image"] == 222 and row["key_question"] == 1012:
    #     #     print(row["original_question_answer"])
    #     # else:
    #     #     continue
    #
    #     # take retrieve_k random contexts index
    #     random.shuffle(nbme_formatted)
    #
    #     # skip if empty
    #     if not row["original_question_answer"]:
    #         logger.warning(f"Empty question answer at index {idx}")
    #         continue
    #     elif pd.isna(row["original_question_answer"]) or pd.isnull(row["original_question_answer"]):
    #         logger.warning(f"Empty question answer at index {idx}")
    #         continue
    #
    #     # create example
    #     examples.append(
    #         dspy.Example(
    #             context=nbme_formatted[:retrieve_k],
    #             question=row["original_question_answer"],
    #             key_question=row["key_question"],
    #             key_image=row["key_image"],
    #         )
    #     )
    #
    # # perform inference and save to df
    # output_list = []
    # for idx, example in tqdm(enumerate(examples), total=len(examples)):
    #     try:
    #         # perform inference to get stage 1 output (revised question answer)
    #         output = trained_module(example)
    #
    #         if "answer" not in output.answer.lower():
    #             logger.info(f"Error in output: {output}")
    #
    #         # try:
    #         #     mcq = MCQ(example=example, prediction=output)
    #         # except Exception:
    #         #     logger.error(f"Error creating MCQ")
    #
    #         # append to output list
    #         output_list.append(
    #             {
    #                 "key_image": example.key_image,
    #                 "key_question": example.key_question,
    #                 "description_question_answer": example.question,
    #                 "revised_question_answer": output.answer,
    #                 # "description": mcq.example_dict.get("description"),
    #                 # "additional_info": mcq.example_dict.get("additional_info"),
    #                 # "original_question": mcq.example_dict.get("question"),
    #                 # "original_question_tokens": mcq.get_tokens(original=True, field="question"),
    #                 # "original_answer": mcq.example_dict.get("correct_answer"),
    #                 # "original_answer_tokens": mcq.get_tokens(original=True, field="correct_answer"),
    #                 # "revised_question": mcq.prediction_dict.get("question"),
    #                 # "revised_question_tokens": mcq.get_tokens(original=False, field="question"),
    #                 # "revised_answer": mcq.prediction_dict.get("correct_answer"),
    #                 # "revised_answer_tokens": mcq.get_tokens(original=False, field="correct_answer"),
    #             }
    #         )
    #
    #         # temp save
    #         if idx % 10 == 0:
    #             logger.info(f"Processed {idx} examples")
    #             output_df = pd.DataFrame(output_list)
    #             output_df.to_csv(
    #                 output_dir.joinpath(f"{model_filepath.stem}_output.csv"),
    #                 index=False,
    #             )
    #
    #     except Exception:
    #         # convert to df and save
    #         output_df = pd.DataFrame(output_list)
    #         output_df.to_csv(
    #             output_dir.joinpath(f"{model_filepath.stem}_output.csv"), index=False
    #         )
    #
    # output_df.to_csv(
    #     output_dir.joinpath(f"{model_filepath.stem}_output_final.csv"), index=False
    # )
    # return

    #####
    # compile rag
    logger.info(
        f"Compiling {module.name} with optimizer {optimizer.name} and model {model.model_name}"
    )
    compiled_rag = optimizer.compile(
        module, trainset=trainset, minibatch_size=len(devset)
    )

    # save compiled rag
    compiled_rag.save(output_dir.joinpath(model_filepath))
    # save history for the last 5 examples
    history_to_jsonl(
        model.lm, output_dir, output_file=f"{model.model_name}_history.jsonl", n=5
    )

    # Evaluate the `compiled_rag` program with the specified metric.
    if not model_filepath.exists():
        logger.error(f"Error saving compiled RAG to {model_filepath}")
        raise FileNotFoundError(f"Error saving compiled RAG to {model_filepath}")

    logger.info(f"Loading compiled RAG from {model_filepath}")
    trained_module = CoTRAG(context=task, num_passages=retrieve_k)
    trained_module.load(model_filepath)
    trained_module.name = trained_module.__class__.__name__

    # evaluate trained rag
    score, results = evaluator(trained_module, metric=eval_metric)
    results_df = pd.DataFrame(results, columns=["example", "prediction", "score"])
    logger.info(f"Compiled RAG score: {score}")
    # save score to yaml
    if results_file.exists():
        results = yaml_loader(results_file)
        if model.model_name not in results:
            results[model.model_name] = {}
        results[model.model_name][module.signature_name][optimizer.name] = float(score)
        yaml_writer(results, results_file)
    else:
        yaml_writer(
            {model.model_name: {module.signature_name: {optimizer.name: float(score)}}},
            results_file,
        )

    # Save the results
    results_df.to_csv(
        output_dir.joinpath(f"{model_filepath.stem}_results.csv"), index=False
    )


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()


# # merge df
#     # read df
#     df = df_loader(Path(DATA_ROOT).joinpath(dataset_name))
#     df_complete = df_loader(Path(DATA_ROOT).joinpath("o1-mini_CoTRAG_SelfAssessRevisedInput_output_processed.csv"))
#     # drop rows where df[["key_image","key_question"]] match df_complete[["key_image","key_question"]]
#     df["key_image_question"] = df.apply(lambda x: f"{x['key_image']}_{x['key_question']}", axis=1)
#     df_complete["key_image_question"] = df_complete.apply(lambda x: f"{x['key_image']}_{x['key_question']}", axis=1)
#
#     idx_overlap = df[df["key_image_question"].isin(df_complete["key_image_question"])].index
#     # drop rows with idx_overlap
#     df = df.drop(idx_overlap)
#
#     # save smaller df
#     df.to_csv(Path(DATA_ROOT).joinpath(dataset_name), index=False)

# ##### merge postbot
#     output_csv = Path("/home/jjn/GitHub/microchat/data/processed").joinpath("o1-mini_CoTRAG_SelfAssessRevisedInput_output_processed_temp.csv")
#     output_df = pd.read_csv(output_csv)
#
#     # llm refined questions
#     llm_revised_df = Path("/home/jjn/GitHub/microchat/data/processed").joinpath("VQA_3_bot_refined - exp_1105_test150_dspy_150_best_5.csv.csv")
#     llm_revised_df = pd.read_csv(llm_revised_df)
#     logger.info(f"Processing {len(output_df)} examples")
#
#     llm_revised_df_subset = llm_revised_df[["key_image", "key_question", 'log_str', 'iterations', 'use_case_r', 'code', 'cost', 'question_key', 'mcqs', 'mcqs_formatted', 'question_postbot', 'choices_postbot', 'question_answer_pass', 'mcq_semantic_pass', 'mcqs_format_pass', 'jeff_notes', 'jeff_code']]
#
#     # inner join to only keep rows with key_image and key_question in both dfs
#     output_df = output_df.merge(llm_revised_df_subset, on=["key_image", "key_question"], how="inner")
#
#     # save new name
#     output_df.to_csv(output_dir.joinpath(f"{model_filepath.stem}_output_processed_postbot.csv"), index=False)


# #####
#     # merge df with other_df, but only keep col "description_question_answer, description, additional_info
#     usecols = ["key_image", "key_question", "description_question_answer", "description", "additional_info", "original_question", "original_answer"]
#     df = df.merge(other_df[usecols], on=["key_image", "key_question"], how="inner")
#     df["original_answer_x"] = df["original_answer_x"].str.strip()
#     df["original_answer_y"] = df["original_answer_y"].str.strip()
#
#     # check if all original_question_y is the same as original_question_x
#     assert df["original_question_x"].equals(df["original_question_y"])
#
#     # drop original_question_x and keep original_question_y
#     df = df.drop(columns=["original_question_x"])


# # #####
# # create col 'naive_question_answer'
# # {'question': 'Why does the resolution of the same cryo-EM image decrease when the sample is exposed to a high cumulative dose of electrons?', 'answer': 'The dosage affects final resolution because as high-energy electrons interact with the atoms in the sample, this causes ionization and breaking of chemical bonds and a consequent rearrangement of chemical bonds. As a result, the fine details are lost and the raw data may not adequately reflect the original structure. So as cumulative dose increases, this will cause more damage to the sample and more of the samples atoms will ionize and the final resolution will be lower.', 'incorrect_answers': ['Higher cumulative doses improve the resolution by enhancing the image contrast, making the structure clearer.', 'The resolution decreases because the increased dose of electrons cools down the sample, causing it to contract and distort.', "High cumulative doses enhance chemical bond formation, leading to a clearer depiction of the sample's structure.", "Cumulative electron dose does not affect resolution; changes in the image clarity are due to the microscope's settings.", 'The resolution weakens because the electrons form a protective layer around the sample, which interferes with image clarity.']}
# # parse llm_response_choices
# df["naive_question"] = df["llm_response_choices"].apply(lambda x: eval(x)["question"])
# df["naive_answer"] = df["llm_response_choices"].apply(lambda x: eval(x)["answer"])
# df["naive_incorrect_answers"] = df["llm_response_choices"].apply(lambda x: eval(x)["incorrect_answers"])
# # join answer with incorrect answers and random permutation
# df["naive_choices"] = df.apply(lambda x: [x["naive_answer"]] + x["naive_incorrect_answers"], axis=1)
# df["naive_choices"] = df["naive_choices"].apply(lambda x: random.sample(x, len(x)))
# df["naive_correct_index"] = df.apply(lambda x: x["naive_choices"].index(x["naive_answer"]), axis=1)
# df["naive_correct_index_char"] = df["naive_correct_index"].apply(lambda x: f"{chr(65 + x)}) ")
#
# # format choices as A) B) C) D) E)
# df["naive_choices_formatted"] = df["naive_choices"].apply(
#     lambda x: "\n".join([f"{chr(65 + i)}) {choice}" for i, choice in enumerate(x)])
# )
#
# # format MCQ
# # Question:
# # ```<question>
# #
# # <options>
# # Correct answer: <answer>
# # ```
# df["naive_question_answer_formatted"] = (
#             "Question:\n```" + df["naive_question"] + "\n\n" + df["naive_choices_formatted"] + "\n\nCorrect Answer: " +
#             df["naive_correct_index_char"] + df["naive_answer"] + "```")
# df.to_csv(input_file, index=False)
