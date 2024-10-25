#!/usr/bin/env python3
"""base_signatures.py in src/microchat/models."""

import dspy

from microchat import MODULE_ROOT
from microchat.fileio.text.readers import yaml_loader
from pathlib import Path

default_instructions = yaml_loader(Path(MODULE_ROOT, "conf", "signatures.yaml"))
base_qa = default_instructions["BasicQA"]


class ReviseQuestion(dspy.Signature):
    """You are an expert in BioMedical AI, helping biologists and computer scientists design a benchmark to test vision-language models' perception and reasoning capabilities. You are helping the benchmark creators take user-submitted questions and convert them into high-quality question stems for multiple-choice questions.

    Your expertise and body of knowledge include a deep familiarity with Bloom's taxonomy and training from the National Board of Medical Examiners on how to write high-quality multiple-choice items to test content knowledge and reasoning. You always state if you are uncertain or unsure how to write a question stem. You are also familiar with the concept of "stem-equity" in multiple-choice questions and are always looking for ways to improve the quality of the question stem.
    """

    # # NBME Guidelines
    # <nbme_mc-test_guidelines>
    #   ## One best answer items
    #   The one-best-answer questions are designed to make explicit that only one option is to be selected. These items are the most widely used multiple-choice item format. They consist of a stem, which most often includes a vignette (eg, a clinical or scientific scenario) and a lead-in question, followed by a series of option choices, with one correct answer and anywhere from three to seven distractors. *The incorrect option choices should be directly related to the lead-in and be homogeneous with the correct answer*. This item describes a situation (in this instance, a patient scenario) and asks the test-taker to indicate the most likely cause of the problem, most likely mechanism of action, most likely subcellular structure, or best next step.
    #
    #   ## General Rules for One-Best-Answer Items
    #   Because test-takers are required to select the single best answer, one-best-answer items must satisfy the following rules (for more detail, see the six rules below):
    #   - Item and option text must be clear and unambiguous. Avoid imprecise phrases such as “is associated with” or “is useful for” or “is important”; words that provide cueing such as “may” or “could be”; and vague terms such as “usually” or “frequently.”
    #   - The lead-in should be closed and focused and ideally worded in such a way that the test-taker can cover the options and guess the correct answer. This is known as the “cover-the-options” rule.
    #   - All options should be homogeneous to be judged as entirely true or false on a single dimension.
    #   - Incorrect options can be partially or wholly incorrect.
    #
    #   ## The Shape of a Good Multiple-Choice Item
    #   A well-constructed one-best-answer item will have a particular format below. A biological or experimental scenario with context may serve as the stem. The possible options are listed below concisely and uniformly. The stem should include all relevant facts; no additional information should be provided in the options.
    #
    #   <question_stem>
    #   <vignette>Provide the minimum necessary context in the vignette, but do not give away the answer.<vignette>
    #   <lead-in>Pose your question here in the lead-in.<lead-in>:
    #   <question_stem>
    #
    #   <options>Insert your answer option set here, making sure it follows the “cover-the-options” rule.<options>
    #
    #   ## Guideline for writing item lead-in
    #   The lead-in should contain a single, clearly formulated question so the test-taker can answer without looking at the options. Satisfying the “cover-the-options” rule is essential to a good question.
    #   </nbme_mc-test_guidelines>
    question = dspy.InputField(desc="The original question submitted by the user.")
    answer = dspy.OutputField(
        desc="An improved question stem according to NBME guidelines."
    )


class ReviseQuestionContext(dspy.Signature):
    """You are an expert in BioMedical AI, helping biologists and computer scientists design a benchmark to test vision-language models' perception and reasoning capabilities. You are helping the benchmark creators take user-submitted questions and convert them into high-quality question stems for multiple-choice questions.

    Your expertise and body of knowledge include a deep familiarity with Bloom's taxonomy and training from the National Board of Medical Examiners on how to write high-quality multiple-choice items to test content knowledge and reasoning. You always state if you are uncertain or unsure how to write a question stem. You are also familiar with the concept of "stem-equity" in multiple-choice questions and are always looking for ways to improve the quality of the question stem.
    """

    context = dspy.InputField(
        desc="NBME guidelines for writing multiple-choice questions."
    )
    question = dspy.InputField(desc="The original question submitted by the user.")
    answer = dspy.OutputField(
        desc="An improved question stem according to NBME guidelines."
    )

class ClassifyBlooms(dspy.Signature):
    """You are an expert in BioMedical AI, helping biologists and computer scientists design a benchmark to test vision-language models' perception and reasoning capabilities. You are helping the benchmark creators take user-submitted questions and convert them into high-quality question stems for multiple-choice questions.

    Your expertise and body of knowledge include a deep familiarity with Bloom's taxonomy and training from the National Board of Medical Examiners on how to write high-quality multiple-choice items to test content knowledge and reasoning. You always state if you are uncertain or unsure how to write a question stem. You are also familiar with the concept of "stem-equity" in multiple-choice questions and are always looking for ways to improve the quality of the question stem.
    """

    context = dspy.InputField(
        desc="Bloom's taxonomy for writing multiple-choice questions."
    )
    question = dspy.InputField(desc="The revised question formatted for a one-best-answer multiple choice.")
    answer = dspy.OutputField(
        desc="The Bloom's taxonomy category for the revised question."
    )

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")