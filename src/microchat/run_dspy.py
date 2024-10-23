#!/usr/bin/env python3
"""run_dspy.py in src/microchat."""
from pathlib import Path
from typing import Optional

import click
from dotenv import find_dotenv
from dotenv import load_dotenv


from loguru import logger

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2
from dspy.evaluate.evaluate import Evaluate
from dsp.utils import deduplicate

from microchat.custom_datasets.dataset_factory import create_dataset
from microchat.models.base_signatures import BasicQA
from microchat.models.model_factory import create_model

# from litellm import cache

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
@click.option(
    "--input-dir", type=click.Path(file_okay=False, exists=True, path_type=Path)
)
@click.option(
    "--output-dir", type=click.Path(file_okay=False, exists=False, path_type=Path)
)
@click.option("--model", type=click.Choice(["gpt-4o-mini"]), default="gpt-4o-mini")
@click.option(
    "--dataset_name", type=click.Choice(["hotpotqa", "scieval"]), default="scieval"
)
@click.option("--random-seed", type=click.INT, default=42)
@click.option("--retrieve-k", type=click.IntRange(3, 10), default=5)
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    model: Optional[str] = "gpt-4o-mini",
    dataset_name: Optional[str] = "scieval",
    random_seed: int = 8675309,
    retrieve_k: int = 5,
    dry_run: bool = False,
) -> None:
    """Docstring."""
    # output_dir = output_dir or input_dir.joinpath("processed")
    # output_dir.mkdir(parents=True, exist_ok=True)

    project_dir = Path(__file__).parents[2]
    logger.add(
        project_dir.joinpath("logs", f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model}")

    if dry_run:
        logger.info("Dry run: no changes will be made.")
        return

    # instantiate model LLM/VLM model
    model = create_model(model)

    # define retrieval model
    # TODO: refactor to use model_registry
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(
        url="http://20.102.90.50:2017/wiki17_abstracts"
    )

    # configure DSPy settings
    dspy.settings.configure(lm=model, rm=colbertv2_wiki17_abstracts)

    # instantiate dataset
    dataset = create_dataset(dataset_name)

    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]

    print(f"{len(trainset)}, {len(devset)}")

    train_example = trainset[0]
    print(f"Question: {train_example.question}")
    print(f"Answer: {train_example.answer}")

    dev_example = devset[18]
    print(f"Question: {dev_example.question}")
    print(f"Answer: {dev_example.answer}")
    print(f"Relevant Wikipedia Titles: {dev_example.gold_titles}")

    print(
        f"For this dataset, training examples have input keys {train_example.inputs().keys()} and label keys {train_example.labels().keys()}"
    )
    print(
        f"For this dataset, dev examples have input keys {dev_example.inputs().keys()} and label keys {dev_example.labels().keys()}"
    )

    # Define the predictor.
    generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    pred = generate_answer(question=dev_example.question)

    # Print the input and the prediction.
    print(f"Question: {dev_example.question}")
    print(f"Predicted Answer: {pred.answer}")

    model.inspect_history(n=1)

    # Define the predictor. Notice we're just changing the class. The signature BasicQA is unchanged.
    generate_answer_with_chain_of_thought = dspy.ChainOfThought(BasicQA)

    # Call the predictor on the same input.
    pred = generate_answer_with_chain_of_thought(question=dev_example.question)

    # Print the input, the chain of thought, and the prediction.
    print(f"Question: {dev_example.question}")
    print(f"Thought: {pred.reasoning.split('.', 1)[1].strip()}")
    print(f"Predicted Answer: {pred.answer}")

    # retrieve
    retrieve = dspy.Retrieve(k=retrieve_k)
    topK_passages = retrieve(dev_example.question).passages

    print(
        f"Top {retrieve.k} passages for question: {dev_example.question} \n",
        "-" * 30,
        "\n",
    )

    for idx, passage in enumerate(topK_passages):
        print(f"{idx+1}]", passage, "\n")

    #
    class GenerateAnswer(dspy.Signature):
        """Answer questions with short factoid answers."""

        context = dspy.InputField(desc="may contain relevant facts")
        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    # Define the module
    class RAG(dspy.Module):
        def __init__(self, num_passages=3):
            super().__init__()

            self.retrieve = dspy.Retrieve(k=num_passages)
            self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

        def forward(self, question):
            context = self.retrieve(question).passages
            prediction = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=prediction.answer)

    # Validation logic: check that the predicted answer is correct.
    # Also check that the retrieved context does actually contain that answer.
    def validate_context_and_answer(example, pred, trace=None):
        answer_EM = dspy.evaluate.answer_exact_match(example, pred)
        answer_PM = dspy.evaluate.answer_passage_match(example, pred)
        return answer_EM and answer_PM

    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

    # Compile!
    compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

    ##
    # Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
    evaluate_on_hotpotqa = Evaluate(
        devset=devset, num_threads=1, display_progress=True, display_table=5
    )

    # Evaluate the `compiled_rag` program with the `answer_exact_match` metric.
    metric = dspy.evaluate.answer_exact_match
    evaluate_on_hotpotqa(compiled_rag, metric=metric)

    ##
    class GenerateSearchQuery(dspy.Signature):
        """Write a simple search query that will help answer a complex question."""

        context = dspy.InputField(desc="may contain relevant facts")
        question = dspy.InputField()
        query = dspy.OutputField()

    class SimplifiedBaleen(dspy.Module):
        def __init__(self, passages_per_hop=3, max_hops=2):
            super().__init__()

            self.generate_query = [
                dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
            ]
            self.retrieve = dspy.Retrieve(k=passages_per_hop)
            self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
            self.max_hops = max_hops

        def forward(self, question):
            context = []

            for hop in range(self.max_hops):
                query = self.generate_query[hop](
                    context=context, question=question
                ).query
                passages = self.retrieve(query).passages
                context = deduplicate(context + passages)

            pred = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=pred.answer)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
