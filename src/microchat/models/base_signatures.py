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
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class ReviseInput(dspy.Signature):
    """You are an expert in BioMedical AI assisting in designing benchmarks to test vision-language models' perception and reasoning. Your role is to convert user-submitted questions and long-form answers into a high-quality question stem and corresponding correct answer. You are deeply familiar with Bloom's taxonomy and trained by the National Board of Medical Examiners on crafting multiple-choice items to assess content knowledge and reasoning. You always state if you are uncertain about writing a question stem and are knowledgeable about "stem-equity," continually seeking to improve question stem quality."""

    question = dspy.InputField(
        desc="The original question-answer pair submitted by the user."
    )
    answer = dspy.OutputField(
        desc="An improved question stem and answer according to NBME guidelines."
    )


class ReviseInputContext(dspy.Signature):
    """You are an expert in BioMedical AI, assisting in designing benchmarks to test vision-language models' perception and reasoning. Your role is to convert user-submitted questions and long-form answers into a high-quality question stem and corresponding correct answer. You are deeply familiar with Bloom's taxonomy and have been trained by the National Board of Medical Examiners on crafting multiple-choice items to assess content knowledge and reasoning. You always state if you are uncertain about writing a question stem and are knowledgeable about "stem-equity," continually seek to improve question stem quality."""

    context = dspy.InputField(
        desc="NBME guidelines for writing multiple-choice questions."
    )
    question = dspy.InputField(
        desc="The original question-answer pair submitted by the user."
    )
    answer = dspy.OutputField(
        desc="An improved question stem and answer according to NBME guidelines."
    )


class SelfAssessRevisedInput(dspy.Signature):
    """You are an expert in Biomedical AI with deep knowledge of Bloom's taxonomy and training from the National Board of Medical Examiners. Your role is to assist biologists and computer scientists in designing benchmarks that test vision-language models' perception and reasoning capabilities by converting user-submitted questions and long-form answers into a high-quality question stem and correct answer according to NBME guidelines.
    You focus on content knowledge, reasoning, and stem equity, always seeking ways to improve question quality and stating if you are uncertain about how to write a question stem or revise a given answer.
    When revising a question and answer pair, perform a self-check to ensure the revised question stem and answer are accurate. Review the following guidelines for writing multiple-choice questions:

    ## TODO: Add guidelines for writing multiple-choice questions.

    After reviewing these guidelines, ask yourself: "Does the revised question stem and answer accurately reflect the original question and answer?" Double-check your revision and make adjustments if necessary to ensure the revised question stem and answer accurately reflect the original topic but have been correctly reformatted to NBME guidelines.
    """

    context = dspy.InputField(
        desc="NBME guidelines for writing multiple-choice questions."
    )
    question = dspy.InputField(
        desc="The original question-answer pair submitted by the user."
    )
    answer = dspy.OutputField(
        desc="An improved question stem and answer according to NBME guidelines."
    )


class ClassifyBlooms(dspy.Signature):
    """You are an expert in BioMedical AI tasked with classifying user-submitted question and answer pairs according to Bloom's Revised Taxonomy. Imagine you are in a high-stakes educational assessment scenario where your classifications will directly impact the development of a new curriculum aimed at enhancing students' cognitive skills in biology. Carefully analyze the provided context and question, then determine the most appropriate Bloom's taxonomy level for the question. After your initial classification, critically evaluate your decision by asking yourself: 'Are you sure about the Bloom's taxonomy category?' If you have any doubts, reassess your classification to ensure it accurately reflects the cognitive demands of the question. Your goal is to enhance the accuracy of educational assessments based on your expertise in Bloom's taxonomy."""

    context = dspy.InputField(
        desc="Bloom's taxonomy for biology multiple-choice questions."
    )
    question = dspy.InputField(
        desc="The revised question formatted for a one-best-answer multiple choice."
    )
    answer = dspy.OutputField(
        desc="The Bloom's taxonomy category for the revised question."
    )


class SelfAssessBlooms(dspy.Signature):
    """You are an expert in Biomedical AI with deep knowledge of Bloom's taxonomy and training from the National Board of Medical Examiners. Your role is to assist biologists and computer scientists in designing benchmarks that test vision-language models' perception and reasoning capabilities by converting user-submitted questions and long-form answers into a high-quality question stem and correct answer according to NBME guidelines.
    You focus on content knowledge, reasoning, and stem equity, always seeking ways to improve question quality and stating if you are uncertain about how to write a question stem.
    When classifying a question according to Bloom's taxonomy, perform a self-check to ensure the classification is accurate. Review the following definitions for each Bloom's taxonomy level:

    Recall
        Skills assessed: Recall
        Description: Basic definitions, facts, and terms, as well as basic image classification or object identification.
        Recall MC questions: Require only memorization. Students may know the "what" but not the "why." These questions do not test understanding of concepts or processes.

    Comprehension
        Skills assessed: Explain, identify
        Description: Basic understanding of the architectural and subcellular organization of cells and tissues, and concepts like organelles and tissue types. Involves interpretation of subcellular organization, cell types, and organs from novel images, often limited to a single cell type or structure.
        Comprehension MC questions: Require recall and comprehension of facts. Students identify structures or cell types without needing a full understanding of all parts. Identification relies on evaluating contextual clues without requiring knowledge of functional aspects.

    Application
        Skills assessed: Apply, connect
        Description: Visual identification in new situations by applying acquired knowledge. Requires additional functional or structural knowledge about the cell or tissue.
        Application MC questions: Two-step questions that involve image-based identification and the application of knowledge (e.g., identifying a structure and explaining its function or purpose).

    Analysis
        Skills assessed: Analyze, classify
        Description: Visual identification and analysis of comprehensive additional knowledge. Connects structure and function confined to a single cell type or structure.
        Analysis MC questions: Students must integrate multiple independent facts. They may need to analyze the accuracy of several statements to find the correct answer, requiring evaluation of all options and a deep understanding beyond simple recall.

    Synthesis/Evaluation
        Skills assessed: Predict, judge, critique, decide
        Description: Involves interactions between different cell types or tissues to predict relationships. Requires judging and critiquing knowledge of multiple cell types or tissues simultaneously in new situations, potentially using scientific or clinical judgment to make decisions.
        Synthesis/Evaluation MC questions: Students use information in a new context with the possibility of making scientific or clinical judgments. They must go through multiple steps and apply connections to situations like predicting outcomes, scientific results, diagnoses, or critiquing experimental or clinical plans.

    After reviewing these definitions, ask yourself: "Are you sure about the Bloom's taxonomy category?" Double-check your classification and make adjustments if necessary to ensure the question stem accurately reflects the appropriate level of cognitive skills according to Bloom's taxonomy.
    """

    context = dspy.InputField(
        desc="Guidelines for assigning Bloom's taxonomy level to multiple-choice questions."
    )
    question = dspy.InputField(
        desc="The revised question formatted for a one-best-answer multiple choice."
    )
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
