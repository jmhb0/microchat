#!/usr/bin/env python3
"""mcq.py in src/microchat/mc_questions.

This module defines the MCQ model using Pydantic to interact with DSPy examples and modules.
It provides methods for prediction, cleaning answers, and extracting multiple-choice options.
"""
import os
import re

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any, Tuple
from loguru import logger
import dspy
import tiktoken as tk
from microchat.metrics.token_metrics import compute_token_metric

from microchat.models.base_signatures import SelfAssessBlooms, CheckSimilar
from microchat.models.dspy_modules import re_blooms_compiled, blooms_dict
from microchat.models.model_factory import create_model
from microchat.utils.process_text import process_blooms, compute_tokens, compute_chars

re_clean_text = re.compile(r"`{1,3}|'{1,3}", re.IGNORECASE)
re_correct_ans = re.compile(r"\(Correct\)$", re.IGNORECASE)
re_true = re.compile(r"(True|Yes|Correct|Right)", re.IGNORECASE)
re_false = re.compile(r"(False|No|Incorrect|Wrong)", re.IGNORECASE)
# re_correct_incorrect = re.compile(r"\((Correct)\)|\((Incorrect)\)", re.IGNORECASE)
re_clean_opt = re.compile(
    r"^\d+\.\s|^\w{1}\.\s|\(Correct\)$|\(Incorrect\)$", re.IGNORECASE
)
re_parse_example = re.compile(
    r"Description of image preparation:\n(?P<quote1>['`]{2,3})(?P<description>.*?)(?P=quote1)\n+"
    r"(Additional information:\n(?P<quote2>['`]{2,3})(?P<additional_info>.*?)(?P=quote2)\n+)?"
    r"Question:\n(?P<quote3>['`]{2,3})(?P<question>.*?)(?P=quote3)\n+"
    r"Answer:\n(?P<quote4>['`]{2,3})(?P<correct_answer>.*?)(?P=quote4)",
    re.IGNORECASE | re.DOTALL,
)

re_parse_options = re.compile(
    r"\n?\*?\*?(?P<option_label>[A-Z])\)\s?\*?\*?\s?(?P<option_text>.*?)(?:\s{2,}|\n+)"
)
re_parse_prediction = re.compile(
    r"(?<=\*\*Question:\*\*\n\n)?(?P<question>.*?)(?=\n\nA\))"  # Capture the question up to 'A)' marking the first option
    r"\n+A\)\s?(?P<option_a>.*?)(?:\s{2,}|\n+)"  # Capture option A with flexible whitespace handling
    r"B\)\s?(?P<option_b>.*?)(?:\s{2,}|\n+)"  # Capture option B
    r"C\)\s?(?P<option_c>.*?)(?:\s{2,}|\n+)"  # Capture option C
    r"D\)\s?(?P<option_d>.*?)(?:\s{2,}|\n+)"  # Capture option D
    r"\n+\*+([Cc]orrect\s)?[Aa]nswer:\*+\s?(?P<correct_option>\(?[A-Da-d])\)\s?(?P<correct_answer>.*)",  # Capture correct answer with flexible capitalization
    re.IGNORECASE | re.DOTALL,  # Allows matching across multiple lines
)
re_parse_prediction_2 = re.compile(
    r"(?:\*?\*?Revised\s+)?Question(?:\s?\d?)?:(?:\*?\*?)?\s*\n+\n*['`]{0,3}(?P<question>.*?)['`]{0,3}"  # Capture "Revised Question" with optional asterisks and backticks
    r"\n+\n*(?:\*?\*?Revised\s+)?Answer(?:\s?\d?)?:(?:\*?\*?)?\s*\n+\n*['`]{0,3}(?P<correct_answer>.*?)['`]{0,3}",  # Capture "Revised Answer" with optional asterisks and backticks
    re.IGNORECASE | re.DOTALL  # Allows matching across multiple lines
)
re_parse_prediction_3 = re.compile(
    r"(?:Question:\s*)?(?P<question>.*?)(?=\n\n(Revised|Correct)\s?Answer:)"  # Optional "Question:" prefix and captures question up to "Answer" prefix
    r"\n\n(?:Revised|Correct)\s?Answer:\s*\n?(?P<correct_answer>.*)",  # Matches "Correct Answer:" or "Revised Answer:" followed by answer text
    re.IGNORECASE
    | re.DOTALL,  # Allows case-insensitive matching and multi-line capture
)
re_parse_prediction_4 = re.compile(
    r"(?<=\*\*Question:\*\*\n\n)?(?P<question>.*?)(?=\n\n(?:[Nn]o|[Yy]es|[Aa]\)))"  # Capture question up to an answer indicator (No, Yes, or A))
    r"\n\n(?:[Nn]o|[Yy]es|A\))\s?(?P<correct_answer>.*)",  # Match answer prefix (No, Yes, or A)) and capture answer text
    re.IGNORECASE | re.DOTALL,  # Allows case-insensitive and multi-line matching
)


DEFAULT_TEACHER = create_model("o1-mini")


class MCQ(BaseModel):
    """
    Pydantic model for handling multiple-choice question processing.
    It accepts an example and a DSPy module to make predictions.
    """

    example: dspy.Example
    prediction: dspy.Example
    tokenizer: Optional[tk.Encoding] = None

    # variables set from processing the example
    tokenizer_name: Optional[str] = Field(None, alias="tokenizer_name")  # set by model
    example_dict: Optional[Dict[str, str]] = {}
    prediction_dict: Optional[Dict[str, Any]] = {}
    metrics: Optional[Dict[str, Any]] = {}
    errors: Optional[int] = 0

    class Config:
        arbitrary_types_allowed = True

    @field_validator("example")
    def validate_example(cls, value):
        if not isinstance(value, dspy.Example):
            raise ValueError("example must be a DSPy Example instance.")
        return value

    @field_validator("tokenizer")
    def validate_tokenizer(cls, value):
        if not isinstance(value, tk.Encoding):
            raise ValueError("tokenizer must be a TikToken Encoding instance.")
        return value

    @staticmethod
    def compute_chars(prompt: str) -> int:
        """Compute the number of characters in a prompt."""
        return compute_chars(prompt)

    @staticmethod
    def compute_tokens(prompt: str, tokenizer: tk.Encoding) -> int:
        """Compute the number of tokens in a prompt."""
        return compute_tokens(prompt, tokenizer)

    def __repr__(self) -> str:
        """An unambiguous string representation of the class instance."""
        return f"MCQ(example={self.example}, prediction={self.prediction})"

    def __str__(self) -> str:
        """An easy-to-read string representation of the dataset class."""
        question = self.example_dict.get("question", "")
        answer = self.example_dict.get("correct_answer", "")
        revised_question = self.prediction_dict.get("question", "")
        revised_answer = self.prediction_dict.get("correct_answer", "")
        output_str = [
            f"Original question: {question}\nOriginal answer: {answer}\n\n",
            f"----------------------------------------\n",
            f"Revised question: {revised_question}Revised answer: {revised_answer}\n",
            f"----------------------------------------\n",
            f"Metrics: {self.metrics}",
        ]

        return "".join(output_str)

    def compute_metrics(
        self, question_key: str = "question", answer_key: str = "correct_answer"
    ) -> Dict[str, float]:
        temp_example = dspy.Example(
            question=self.example_dict.get(question_key),
            answer=self.example_dict.get(answer_key),
        )
        temp_pred = dspy.Example(
            question=self.prediction_dict.get(question_key,""),
            answer=self.prediction_dict.get(answer_key,""),
        )

        # check for exact match of the answer
        match = False
        try:
            match = dspy.evaluate.answer_exact_match(temp_example, temp_pred)
        except Exception as e:
            logger.error(f"Error in exact match evaluation: {e}")

        # exit early if exact match is found
        if match:
            return {
                "similarity": float(match),
                "formatted": float(match),
                "extraneous": 1 - float(match),
            }

        # check for semantic similarity of the question
        context = self.example_dict.get("description")
        if addtl_info := self.example_dict.get("additional_info"):
            context += f"\n\nAdditional information: {addtl_info}"

        question_str = (
            f"Original question: {temp_example.question}\nOriginal answer: {temp_example.answer}\n\n"
            f"----------------------------------------\n"
            f"Revised question: {temp_pred.question}Revised answer: {temp_pred.answer}\n"
        )
        with dspy.settings.context(lm=DEFAULT_TEACHER.lm):
            result = dspy.ChainOfThought(CheckSimilar)(
                context=context, question=question_str
            )

        # clean text outputs
        similarity = re_clean_text.sub("", result.similarity).strip()
        similarity = re_true.sub("1", similarity)
        formatted = re_clean_text.sub("", result.formatted).strip()
        formatted = re_true.sub("1", formatted)
        extraneous = re_clean_text.sub("", result.extraneous).strip()
        extraneous = re_true.sub("1", extraneous)
        try:
            similarity = float(eval(similarity))
            formatted = float(eval(formatted))
            extraneous = float(eval(extraneous))
        except ValueError as e:
            logger.error(f"Error in converting metrics to float: {e}")
            similarity = 0
            formatted = 0
            extraneous = 0

        return {
            "similarity": float(similarity),
            "formatted": float(formatted),
            "extraneous": 1 - float(extraneous),
        }

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for the MCQ model.
        """
        lm_name = getattr(dspy.settings.lm, "model_name", None)
        lm_prefix = getattr(dspy.settings.lm, "model_prefix", None)
        if self.tokenizer is None:
            # set tokenizer name
            if "o1" in lm_prefix:
                self.tokenizer_name: str = "o200k_base"
            else:
                self.tokenizer_name: str = tk.encoding_name_for_model(lm_prefix)

            self.tokenizer = tk.get_encoding(self.tokenizer_name)

        # process the example
        example_match = re_parse_example.search(self.example.question)
        if example_match is None:
            self.errors += 1
            logger.warning("Example does not match the expected format.")
        else:
            self.example_dict = example_match.groupdict()

        # process the prediction
        pred_match = re_parse_prediction.search(self.prediction.answer)
        pred_match = (
            re_parse_prediction_2.search(self.prediction.answer)
            if pred_match is None
            else pred_match
        )
        pred_match = (
            re_parse_prediction_3.search(self.prediction.answer)
            if pred_match is None
            else pred_match
        )
        pred_match = (
            re_parse_prediction_4.search(self.prediction.answer)
            if pred_match is None
            else pred_match
        )
        if pred_match is None:
            self.errors += 1
            logger.warning("Prediction does not match the expected format.")
        else:
            prediction_dict = pred_match.groupdict()
            # strip leading \* from the question
            for key in ["question", "correct_answer"]:
                if prediction_dict.get(key):
                    prediction_dict[key] = prediction_dict[key].strip("*").strip()
            # clean_text
            for key in ["question", "correct_answer"]:
                if prediction_dict.get(key):
                    prediction_dict[key] = re_clean_text.sub("", prediction_dict[key])
            self.prediction_dict = prediction_dict

        # extract the question, options, and correct answer from the prediction
        pred_question = self.prediction_dict.get("question")
        pred_options = [
            self.prediction_dict.get(f"option_{char}")
            for char in "abcdefghijklmnop"
            if self.prediction_dict.get(f"option_{char}")
        ]
        pred_answer = self.prediction_dict.get("correct_answer")
        pred_option_correct = self.prediction_dict.get("correct_option")
        pred_correct_index = -1
        if pred_answer in pred_options:
            pred_correct_index = pred_options.index(pred_answer)

        # check if pred_answer tokens are longer than the original answer
        ans_token_metric = 0
        if self.example_dict.get("correct_answer") and pred_answer:
            orig_tokens = self.compute_tokens(
                self.example_dict.get("correct_answer"), self.tokenizer
            )
            pred_tokens = self.compute_tokens(pred_answer, self.tokenizer)
            ans_token_ratio = pred_tokens / orig_tokens
            ans_token_metric = compute_token_metric(orig_tokens, pred_tokens, k=0.5)

            if ans_token_ratio >= 1.5:
                logger.warning(
                    f"Predicted answer longer: {orig_tokens} vs. {pred_tokens}"
                )
                logger.warning(
                    f"Original answer: {self.example_dict.get('correct_answer')}"
                )
                logger.warning(f"Predicted answer: {pred_answer}")

        # compare token difference example and prediction
        token_diff = {}
        for key in ["question", "correct_answer"]:
            if key in self.example_dict and key in self.prediction_dict:
                example_tokens = self.compute_tokens(
                    self.example_dict.get(key, ""), self.tokenizer
                )
                pred_tokens = self.compute_tokens(
                    self.prediction_dict.get(key, ""), self.tokenizer
                )
                token_diff[key] = abs(example_tokens - pred_tokens)

        # compute the number of tokens in the options and correct answer
        option_token_ratio = 1  # if no pred_options, don't penalize the model
        if pred_options:
            option_tokens = [
                self.compute_tokens(option, self.tokenizer) for option in pred_options
            ]
            correct_tokens = option_tokens[pred_correct_index] or 1
            incorrect_tokens = [
                tokens
                for i, tokens in enumerate(option_tokens)
                if i != pred_correct_index
            ]
            token_ratio = np.divide(np.array(incorrect_tokens), correct_tokens)

            # compute metric for token ratio, want to have ratio near 1
            option_token_ratio = np.mean(token_ratio)

        #
        try:
            self.metrics = self.compute_metrics(
                question_key="question", answer_key="correct_answer"
            )
        except Exception as e:
            logger.error(f"Error in computing metrics: {e}")
            self.metrics = {
                "similarity": 0,
                "formatted": 0,
                "extraneous": 0,
            }

        self.metrics["option_token_ratio"] = option_token_ratio
        self.metrics["answer_token_metric"] = ans_token_metric
        self.metrics["errors"] = 1 - (self.errors / 2)


class Blooms(BaseModel):
    """
    Pydantic model for handling multiple-choice question processing.
    It accepts an example and a DSPy module to make predictions.
    """

    example: dspy.Example
    module: Any
    tokenizer: Optional[tk.Encoding] = None
    teacher_model: Optional[dspy.LM] = None

    #
    blooms_level: Optional[int] = None
    blooms_name: Optional[str] = None
    blooms_confidence: Optional[float] = None
    blooms_source: Optional[str] = None
    blooms_reasoning: Optional[str] = None
    context: Optional[List[str]] = None
    self_check_question: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator("example")
    def validate_example(cls, value):
        if not isinstance(value, dspy.Example):
            raise ValueError("example must be a DSPy Example instance.")
        return value

    @field_validator("module")
    def validate_module(cls, value):
        if not isinstance(value, dspy.Module):
            raise ValueError("module must be a DSPy Module instance.")
        return value

    @field_validator("tokenizer")
    def validate_tokenizer(cls, value):
        if not isinstance(value, tk.Encoding):
            raise ValueError("tokenizer must be a TikToken Encoding instance.")
        return value

    @staticmethod
    def compute_chars(prompt: str) -> int:
        """Compute the number of characters in a prompt."""
        return compute_chars(prompt)

    @staticmethod
    def compute_tokens(prompt: str, tokenizer: tk.Encoding) -> int:
        """Compute the number of tokens in a prompt."""
        return compute_tokens(prompt, tokenizer)

    @staticmethod
    def _process_answer(
        answer: str, reference_dict: Optional[dict] = blooms_dict
    ) -> Tuple[int, str]:
        return process_blooms(answer, reference_dict)

    def predict(self) -> dspy.Prediction:
        """
        Predict the answer using the DSPy module.
        """
        # if os.getenv("DEBUG", False):
        #     logger.debug(f"Predicting answer for question: {self.example.question}")

        try:
            response = self.module(self.example.question)
            if response is None:
                logger.error(f"Prediction failed for question: {self.example.question}")

            if getattr(response, "reasoning", None) is None:
                response.reasoning = "No reasoning provided."
            else:
                response.reasoning = response.reasoning

            return response
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise e

    def __repr__(self) -> str:
        """An unambiguous string representation of the class instance."""
        return f"Blooms(\n\texample={self.example},\n\tmodule={self.module}\n)"

    def __str__(self) -> str:
        """An easy-to-read string representation of the dataset class."""
        return f"{self.example.question}\nBloom's: {self.blooms_name.capitalize()} (level {self.blooms_level})"

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for the MCQ model.
        """
        # process GT example to get the ground truth blooms name and level
        if isinstance(self.example.blooms_source, str):
            gt_model = self.example.blooms_source.split("&")  # "CustomGPT"
        else:
            gt_model = ["CustomGPT"]

        gt_model = [elem.strip() for elem in gt_model]
        gt_level, gt_bloom = self._process_answer(self.example.answer, blooms_dict)
        # predict blooms the example
        response = self.predict()

        # set the context for the response
        # note that the context is manually set to be a curated list of Bloom's
        # or NBME reference information.
        if getattr(response, "context", None) is None:
            self.context = getattr(dspy.settings, "context", None)

        # process the response
        init_model = dspy.settings.lm.model_name
        init_level, init_bloom = self._process_answer(response.answer, blooms_dict)
        #
        self.self_check_question = (
            "Multiple LLM models evaluated the Bloom's Taxonomy level of the following  multiple choice question:\n"
            f"{self.example.question}\n"
            f"One model predicted '{gt_bloom.capitalize()}' (Level {gt_level}) with the following reasoning: {self.example.blooms_reasoning}\n"
            f"A second model predicted '{init_bloom.capitalize()}' (Level {init_level}) with the following reasoning: {response.reasoning}\n\n"
            "# Independent Assessment of Bloom's Taxonomy Level\n"
            "Provide an independent assessment of the most appropriate Bloom's Taxonomy level for the question below. Explain whether you agree or disagree with previous predictions, and why."
            "When evaluating between Comprehension (Level 2) and Application (Level 3) or higher levels, consider:"
            "  - Does the question involve straightforward identification without broader context?"
            "  - Does it require advanced or non-obvious identification?"
            "  - Does it involve application or connection to a broader context?"
            "  - Is it applied to a new or challenging setting?"
            "  - Does it require deep understanding or drawing conclusions that are not obvious?\n"
            f"{self.example.question}\n"
            f"Bloom's:"
        )
        if self.teacher_model is None:
            # model is LLModel class with dspy.LM in model.lm
            self.teacher_model = create_model("o1-mini").lm

        # change context manager to allow self-assessment by a teacher model
        # logger.debug(f"Original model: {dspy.settings.lm.model_name}")
        with dspy.settings.context(lm=self.teacher_model):
            # logger.debug(f"Model: {dspy.settings.lm.model_name}")
            rev_model = dspy.settings.lm.model_name
            assess_response = dspy.ChainOfThought(SelfAssessBlooms)(
                question=self.self_check_question, context=self.context
            )

        # logger.debug(f"Restored model: {dspy.settings.lm.model_name}")
        rev_level, rev_bloom = self._process_answer(assess_response.answer, blooms_dict)

        #
        if gt_level == init_level and gt_level == rev_level:
            # best case scenario, all cases match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 1.0
            self.blooms_source = " & ".join(gt_model + [init_model, rev_model])
            self.blooms_reasoning = assess_response.reasoning
        elif gt_level == init_level:
            # if the ground truth and initial levels match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join(gt_model + [init_model])
            self.blooms_reasoning = response.reasoning
        elif gt_level == rev_level:
            # if the ground truth and self-assessment levels match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join(gt_model + [rev_model])
            self.blooms_reasoning = assess_response.reasoning
        elif rev_level == init_level:
            # if the self-assessment and initial levels match, use the self-assessment level
            self.blooms_level = init_level
            self.blooms_name = init_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join([init_model, rev_model])
            self.blooms_reasoning = assess_response.reasoning
        else:
            # if none of the levels match, use the ground truth level
            logger.error(
                f"Ground truth, initial prediction, and self-assessment levels do not match. {gt_level} != {init_level} != {rev_level}"
            )
            self.blooms_confidence = 0

        # rename self.blooms_name to a standard name
        rename_dict = {
            "Recall": "Recall",
            "Remember": "Recall",
            "Memorize": "Recall",
            "Knowledge": "Recall",
            "Comprehension": "Comprehension",
            "Comprehend": "Comprehension",
            "Understand": "Comprehension",
            "Apply": "Application",
            "Applying": "Application",
            "Analyze": "Analysis",
            "Analyzing": "Analysis",
            "Evaluate": "Evaluation",
            "Evaluating": "Evaluation",
            "Synthesis": "Synthesis",
            "Synthesizing": "Synthesis",
        }
        self.blooms_name = rename_dict.get(self.blooms_name, self.blooms_name)


class MCQTopic(BaseModel):
    """
    Pydantic model for handling multiple-choice question processing.
    It accepts an example and a DSPy module to make predictions.
    """

    example: dspy.Example
    module: Any
    tokenizer: Optional[tk.Encoding] = None
    teacher_model: Optional[dspy.LM] = None

    #
    topic_name: Optional[str] = None
    topic_confidence: Optional[float] = None
    context: Optional[List[str]] = None
    self_check_question: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator("example")
    def validate_example(cls, value):
        if not isinstance(value, dspy.Example):
            raise ValueError("example must be a DSPy Example instance.")
        return value

    @field_validator("module")
    def validate_module(cls, value):
        if not isinstance(value, dspy.Module):
            raise ValueError("module must be a DSPy Module instance.")
        return value

    @field_validator("tokenizer")
    def validate_tokenizer(cls, value):
        if not isinstance(value, tk.Encoding):
            raise ValueError("tokenizer must be a TikToken Encoding instance.")
        return value

    @staticmethod
    def compute_chars(prompt: str) -> int:
        """Compute the number of characters in a prompt."""
        return len(prompt)

    @staticmethod
    def compute_tokens(prompt: str, tokenizer: tk.Encoding) -> int:
        """Compute the number of tokens in a prompt."""
        return len(tokenizer.encode(prompt))

    # @staticmethod
    # def _process_answer(answer: str, reference_dict: dict) -> Tuple[int, str]:
    #     # extract the blooms level from the response
    #     topic_name = None
    #
    #     if match := re_blooms_compiled.search(answer):
    #         blooms_name = match.group().lower()
    #         # find the level of the blooms taxonomy from blooms_dict
    #         blooms_level = next(
    #             level for level, names in reference_dict.items() if blooms_name in names
    #         )
    #     else:
    #         logger.warning(f"Bloom's taxonomy level found in answer: {answer}")
    #
    #     return blooms_level, blooms_name

    def predict(self) -> dspy.Prediction:
        """
        Predict the answer using the DSPy module.
        """
        # if os.getenv("DEBUG", False):
        #     logger.debug(f"Predicting answer for question: {self.example.question}")

        try:
            return self.module(self.example.question)
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise e

    def __repr__(self) -> str:
        """An unambiguous string representation of the class instance."""
        return f"MCQTopic(\n\texample={self.example},\n\tmodule={self.module}\n)"

    def __str__(self) -> str:
        """An easy-to-read string representation of the dataset class."""
        return f"{self.example.question}\nTopic: {self.topic_name.capitalize()}"

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for the MCQ model.
        """
        # process GT example to get the ground truth blooms name and level
        gt_model = self.example.blooms_source.split("&")  # "CustomGPT"
        gt_model = [elem.strip() for elem in gt_model]
        gt_level, gt_bloom = self._process_answer(self.example.answer, blooms_dict)
        # predict blooms the example
        response = self.predict()
        if response is None:
            logger.error(f"Prediction failed for question: {self.example.question}")

        # set the context for the response
        # note that the context is manually set to be a curated list of Bloom's
        # or NBME reference information.
        self.context = response.context

        # process the response
        init_model = dspy.settings.lm.model_name
        init_level, init_bloom = self._process_answer(response.answer, blooms_dict)
        #
        self.self_check_question = (
            "Multiple LLM models evaluated the Bloom's Taxonomy level of the following  multiple choice question:\n"
            f"{self.example.question}\n"
            f"One model predicted '{gt_bloom.capitalize()}' (Level {gt_level}) with the following reasoning: {self.example.blooms_reasoning}\n"
            f"A second model predicted '{init_bloom.capitalize()}' (Level {init_level}) with the following reasoning: {response.reasoning}\n\n"
            "# Independent Assessment of Bloom's Taxonomy Level\n"
            "Provide an independent assessment of the most appropriate Bloom's Taxonomy level for the question below. Explain whether you agree or disagree with previous predictions, and why."
            "When evaluating between Comprehension (Level 2) and Application (Level 3) or higher levels, consider:"
            "  - Does the question involve straightforward identification without broader context?"
            "  - Does it require advanced or non-obvious identification?"
            "  - Does it involve application or connection to a broader context?"
            "  - Is it applied to a new or challenging setting?"
            "  - Does it require deep understanding or drawing conclusions that are not obvious?\n"
            f"{self.example.question}\n"
            f"Bloom's:"
        )
        if self.teacher_model is None:
            # model is LLModel class with dspy.LM in model.lm
            self.teacher_model = create_model("o1-mini").lm

        # change context manager to allow self-assessment by a teacher model
        # logger.debug(f"Original model: {dspy.settings.lm.model_name}")
        with dspy.settings.context(lm=self.teacher_model):
            # logger.debug(f"Model: {dspy.settings.lm.model_name}")
            rev_model = dspy.settings.lm.model_name
            assess_response = dspy.ChainOfThought(SelfAssessBlooms)(
                question=self.self_check_question, context=response.context
            )

        # logger.debug(f"Restored model: {dspy.settings.lm.model_name}")
        rev_level, rev_bloom = self._process_answer(assess_response.answer, blooms_dict)

        #
        if gt_level == init_level and gt_level == rev_level:
            # best case scenario, all cases match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 1.0
            self.blooms_source = " & ".join(gt_model + [init_model, rev_model])
            self.blooms_reasoning = assess_response.reasoning
        elif gt_level == init_level:
            # if the ground truth and initial levels match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join(gt_model + [init_model])
            self.blooms_reasoning = response.reasoning
        elif gt_level == rev_level:
            # if the ground truth and self-assessment levels match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join(gt_model + [rev_model])
            self.blooms_reasoning = assess_response.reasoning
        elif rev_level == init_level:
            # if the self-assessment and initial levels match, use the self-assessment level
            self.blooms_level = init_level
            self.blooms_name = init_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join([init_model, rev_model])
            self.blooms_reasoning = assess_response.reasoning
        else:
            # if none of the levels match, use the ground truth level
            logger.error(
                f"Ground truth, initial prediction, and self-assessment levels do not match. {gt_level} != {init_level} != {rev_level}"
            )
            self.blooms_confidence = 0

        # rename self.blooms_name to a standard name
        rename_dict = {
            "Recall": "Recall",
            "Remember": "Recall",
            "Memorize": "Recall",
            "Knowledge": "Recall",
            "Comprehension": "Comprehension",
            "Comprehend": "Comprehension",
            "Understand": "Comprehension",
            "Apply": "Application",
            "Applying": "Application",
            "Analyze": "Analysis",
            "Analyzing": "Analysis",
            "Evaluate": "Evaluation",
            "Evaluating": "Evaluation",
            "Synthesis": "Synthesis",
            "Synthesizing": "Synthesis",
        }
        self.blooms_name = rename_dict.get(self.blooms_name, self.blooms_name)
