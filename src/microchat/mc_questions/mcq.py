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
import tiktoken_ext
from pprint import pprint

from microchat.models.base_signatures import SelfAssessBlooms
from microchat.models.dspy_modules import re_blooms_compiled, blooms_dict
from microchat.models.model_factory import create_model

re_clean_text = re.compile(r"`{1,3}|")
re_correct_ans = re.compile(r"\(Correct\)$", re.IGNORECASE)
# re_correct_incorrect = re.compile(r"\((Correct)\)|\((Incorrect)\)", re.IGNORECASE)
re_clean_opt = re.compile(
    r"^\d+\.\s|^\w{1}\.\s|\(Correct\)$|\(Incorrect\)$", re.IGNORECASE
)


class MCQ(BaseModel):
    """
    Pydantic model for handling multiple-choice question processing.
    It accepts an example and a DSPy module to make predictions.
    """

    example: dspy.Example
    module: dspy.Module
    tokenizer: Optional[tk.Encoding] = None

    # variables set from processing the example
    context: Optional[List[str]] = None
    question: Optional[str] = Field(None, alias="question", min_length=24)
    answer: Optional[str] = Field(None, alias="answer", min_length=1)
    options: Optional[List[str]] = None
    question_tokens: Optional[int] = None
    options_tokens: Optional[List[int]] = None
    correct_tokens: Optional[int] = None

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

    def predict(self) -> dspy.Prediction:
        """
        Predict the answer using the DSPy module.
        """
        logger.debug(f"Predicting answer for question: {self.example.question}")
        try:
            return self.module(self.example.question)
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise e

    @classmethod
    def clean_pred_answer(cls, pred: dspy.Prediction) -> Dict:
        """
        Clean the prediction answer, extract the question, options, and correct answer.
        """
        logger.debug("Cleaning the prediction and extracting options.")

        response = pred.answer

        # Extract the question from the response
        question = cls._extract_question(response)

        # Extract and clean the predicted answer from the response
        answer = cls._extract_answer(response)

        # Extract and clean multiple choice options
        mc_options, correct_incorrect = cls._extract_options(response)

        if answer is None or answer not in mc_options:
            logger.error(f"Answer: {answer} not found in the options.")
            raise ValueError("Predicted answer not found in the options.")

        # Find the correct answer from the multiple choice options
        correct_answer = mc_options[correct_incorrect.index(True)]

        # Validate if the answer matches the correct answer
        cls._validate_answer(answer, correct_answer)

        # shuffle options
        mc_options = list(mc_options)
        np.random.shuffle(mc_options)

        return {
            "context": pred.context,
            "question": question,
            "answer": answer,
            "options": mc_options,
        }

    @staticmethod
    def _extract_question(response: str) -> str:
        """
        Extract and clean the revised question from the response.
        """
        try:
            question = response.split("Answer:")[0].strip()
        except IndexError as e:
            logger.error("Answer section missing from response.")
            raise ValueError("Answer section is missing.") from e

        # split, assume newline after 'Question:\n```'
        question = question.split("\n")[-1].strip()

        return re_clean_text.sub("", question).strip()

    @staticmethod
    def _extract_answer(response: str) -> str:
        """
        Extract and clean the predicted answer from the response.
        """
        try:
            answer = response.split("Answer:")[1].strip().split("\nOptions")[0]
        except IndexError as e:
            logger.error("Answer section missing from response.")
            raise ValueError("Answer section is missing.") from e

        return re_clean_text.sub("", answer).strip()

    @staticmethod
    def _extract_options(response: str) -> tuple[tuple[str, ...], tuple[bool, ...]]:
        """
        Extract and clean the multiple choice options from the response.
        """
        try:
            mc_options = response.split("Options:")[1].strip().split("\n")
        except IndexError as e:
            logger.error("Options section missing from response.")
            raise ValueError("Options section is missing.") from e

        mc_options = tuple(re_clean_text.sub("", option) for option in mc_options)
        correct_incorrect = tuple(
            bool(re_correct_ans.search(option)) for option in mc_options
        )
        # one correct answer
        if sum(correct_incorrect) != 1:
            logger.error("Multiple correct answers found.")
            raise ValueError("Multiple correct answers found.")

        # clean mc_options to remove prefix and (Correct) or (Incorrect)
        mc_options = tuple(
            re_clean_opt.sub("", option).strip() for option in mc_options
        )

        if len(mc_options) != len(correct_incorrect):
            logger.error("Options and correct/incorrect flags do not match.")
            raise ValueError("Options and correct/incorrect flags do not match.")

        return mc_options, correct_incorrect

    @staticmethod
    def _validate_answer(answer: str, correct_answer: Optional[str]) -> None:
        """
        Validate if the predicted answer matches the correct answer.
        """
        if answer != correct_answer:
            logger.error(
                f"Answer: {answer} does not match Correct Answer: {correct_answer}"
            )
            raise ValueError("Predicted answer does not match the correct answer.")

    def __repr__(self) -> str:
        """An unambiguous string representation of the class instance."""
        return f"MCQ(example={self.example}, module={self.module})"

    def __str__(self) -> str:
        """An easy-to-read string representation of the dataset class."""
        answer_index = self.options.index(self.answer) + 1
        output_str = [f"({self.question_tokens} tkns) Question: {self.question}\n"]
        output_str += [
            f"({tokens} tkns) {i+1}. {option} \n"
            for i, (option, tokens) in enumerate(zip(self.options, self.options_tokens))
        ]
        output_str += "----------------------------------------\n"
        output_str += (
            f"({self.correct_tokens} tkns) Answer: {answer_index}. {self.answer}\n"
        )

        return "".join(output_str)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for the MCQ model.
        """
        if self.tokenizer is None:
            self.tokenizer = tk.get_encoding(dspy.settings.lm.model_prefix)

        # process the example
        response = self.predict()
        cleaned_response = self.clean_pred_answer(response)

        #
        self.context = cleaned_response["context"]
        self.question = cleaned_response["question"]
        self.answer = cleaned_response["answer"]
        self.options = cleaned_response["options"]

        # find index of the correct answer
        correct_index = self.options.index(self.answer)

        # compute the number of chars and tokens in the question and options
        self.question_tokens = self.compute_tokens(self.question, self.tokenizer)
        self.options_tokens = [
            self.compute_tokens(option, self.tokenizer) for option in self.options
        ]
        self.correct_tokens = self.options_tokens[correct_index]

        # compute ratio of tokens in the correct answer to the tokens for all options
        token_ratio = np.divide(np.array(self.options_tokens), self.correct_tokens)


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
        return len(prompt)

    @staticmethod
    def compute_tokens(prompt: str, tokenizer: tk.Encoding) -> int:
        """Compute the number of tokens in a prompt."""
        return len(tokenizer.encode(prompt))

    @staticmethod
    def _process_answer(answer: str, reference_dict: dict) -> Tuple[int, str]:
        # extract the blooms level from the response
        blooms_name = None
        blooms_level = None
        if match := re_blooms_compiled.search(answer):
            blooms_name = match.group().lower()
            # find the level of the blooms taxonomy from blooms_dict
            blooms_level = next(
                level for level, names in reference_dict.items() if blooms_name in names
            )
        else:
            logger.warning(f"Bloom's taxonomy level found in answer: {answer}")

        return blooms_level, blooms_name

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
        return f"Blooms(\n\texample={self.example},\n\tmodule={self.module}\n)"

    def __str__(self) -> str:
        """An easy-to-read string representation of the dataset class."""
        return f"{self.example.question}\nBloom's: {self.blooms_name.capitalize()} (level {self.blooms_level})"

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
