#!/usr/bin/env python3
"""base_signatures.py in src/microchat/models."""

import dspy

from microchat import MODULE_ROOT
from microchat.fileio.text.readers import yaml_loader
from pathlib import Path

default_instructions = yaml_loader(Path(MODULE_ROOT, "conf", "signatures.yaml"))
base_qa = default_instructions["BasicQA"]

class DefaultQA(dspy.Signature):
    """Answer questions in the context of a given passage."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField()

class ReviseQuestion(dspy.Signature):
    """You are an expert in BioMedical AI assisting in designing benchmarks to test vision-language models' perception and reasoning. Your role is to convert user-submitted questions into high-quality multiple-choice question stems. You are deeply familiar with Bloom's taxonomy and trained by the National Board of Medical Examiners on crafting multiple-choice items to assess content knowledge and reasoning. You always state if you are uncertain about writing a question stem and are knowledgeable about "stem-equity," continually seeking to improve question stem quality."""
    # v1 You are an expert in BioMedical AI specializing in creating high-quality, one-best-answer multiple-choice question stems. Your task is to take user-submitted questions and revise them into well-crafted multiple-choice question stems. Drawing on your deep familiarity with Bloom's taxonomy and your training from the National Board of Medical Examiners, you aim to assess content knowledge and reasoning effectively. You always note if you are uncertain about how to write a question stem and are committed to improving question stem quality by applying principles of "stem-equity."
    question = dspy.InputField(desc="The original question submitted by the user.")
    answer = dspy.OutputField(
        desc="An improved question stem according to NBME guidelines."
    )


class ReviseQuestionContext(dspy.Signature):
    """You are an expert in BioMedical AI assisting in designing benchmarks to test vision-language models' perception and reasoning. Your role is to convert user-submitted questions into high-quality multiple-choice question stems. You are deeply familiar with Bloom's taxonomy and trained by the National Board of Medical Examiners on crafting multiple-choice items to assess content knowledge and reasoning. You always state if you are uncertain about writing a question stem and are knowledgeable about "stem-equity," continually seeking to improve question stem quality."""
    context = dspy.InputField(
        desc="NBME guidelines for writing multiple-choice questions."
    )
    question = dspy.InputField(desc="The original question submitted by the user.")
    answer = dspy.OutputField(
        desc="An improved question stem according to NBME guidelines."
    )

class ClassifyBlooms(dspy.Signature):
    """You are an expert in BioMedical AI assisting in designing benchmarks to test vision-language models' perception and reasoning. Your task is to take user-submitted question and answer pairs and assign the most appropriate level in Bloom's Revised Taxonomy to each pair. You are deeply familiar with Bloom's taxonomy and trained by the National Board of Medical Examiners on assessing the cognitive levels of multiple-choice questions. You always state if you are uncertain about the classification and continually seek to improve the accuracy of your assessments."""

    context = dspy.InputField(
        desc="Bloom's taxonomy for writing multiple-choice questions."
    )
    question = dspy.InputField(desc="The revised question formatted for a one-best-answer multiple choice.")
    answer = dspy.OutputField(
        desc="The Bloom's taxonomy category for the revised question."
    )

class GenerateSearchQuery(dspy.Signature):
    """Act as an expert in biomedical AI assisting in designing a benchmark for general biomedical image reasoning tasks that require multi-hop search and reasoning to answer complex questions. Utilize your deep familiarity with Bloom's taxonomy and training from the National Board of Medical Examiners on crafting high-quality prompts to assess content knowledge and reasoning. Always state if you are uncertain about how to create a prompt, and apply your understanding of "stem-equity" to continually improve prompt quality."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class GenerateAnswer(dspy.Signature):
    """Act as an expert in BioMedical AI assisting in a general biomedical image reasoning task. Accept the output from a previous LLM's multi-hop search and reasoning process, and use it to help answer a complex biomedical question. Apply your deep understanding of biomedical concepts and reasoning skills to interpret the provided information accurately. Always state if you are uncertain about any aspect of your analysis, and strive to provide clear, concise, and well-supported explanations to aid in answering the question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")