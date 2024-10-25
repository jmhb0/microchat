#!/usr/bin/env python3
"""dspy_modules.py in src/microchat/models."""

from pathlib import Path
from typing import List

from loguru import logger

import dspy


from microchat import MODULE_ROOT

from microchat.fileio.text.readers import yaml_loader
from microchat.models.base_signatures import ReviseQuestion, ReviseQuestionContext, ClassifyBlooms, GenerateSearchQuery

context = yaml_loader(Path(MODULE_ROOT, "conf", "question_context.yaml"))


# Define the module
class CoTRAG(dspy.Module):
    def __init__(self, num_passages=5, **kwargs):
        super().__init__()

        self.context = None
        self.retrieve = None
        if kwargs.get("context") == "nbme":
            signature: dspy.Signature = ReviseQuestionContext
            self.context = context["nbme"]
            # convert to List[str] with each passage key.strip().replace("_", " ").capitalize() | value.strip()
            self.context = [
                f"{k.strip().replace('_', ' ').capitalize()}: {v.strip()}"
                for k, v in self.context.items()
            ]
        elif kwargs.get("context") == "blooms":
            signature: dspy.Signature = ClassifyBlooms
            self.context = context["blooms"]
            # convert to List[str] with each passage key.strip().replace("_", " ").capitalize() | value.strip()
            self.context = [
                f"{k.strip().replace('_', ' ').capitalize()}: {v.strip()}"
                for k, v in self.context.items()
            ]
        else:
            self.retrieve = dspy.Retrieve(k=num_passages)

        self.generate_answer = dspy.ChainOfThought(signature)
        self.kwargs = kwargs

    def forward(self, question):
        # get fixed context if provided
        if self.context is None:
            self.context = self.retrieve(question).passages
        else:
            context = self.context

        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


class CoTMultiHopRAG(dspy.Module):
    """Simplified Baleen (Khattab et al., 2021) module for multi-hop reasoning."""
    def __init__(self, num_passages=5, passages_per_hop: int = 3, max_hops:int = 3, **kwargs):
        super().__init__()

        self.context: List[str] = None
        self.retrieve = None
        self.generate_query: dspy.Signature = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.max_hops = 2
        if kwargs.get("context") == "nbme":
            signature: dspy.Signature = ReviseQuestionContext
            self.context = context["nbme"]
            # convert to List[str] with each passage key.strip().replace("_", " ").capitalize() | value.strip()
            self.context = [
                f"{k.strip().replace('_', ' ').capitalize()}: {v.strip()}"
                for k, v in self.context.items()
            ]
        elif kwargs.get("context") == "blooms":
            signature: dspy.Signature = ClassifyBlooms
            self.context = context["blooms"]
            # convert to List[str] with each passage key.strip().replace("_", " ").capitalize() | value.strip()
            self.context = [
                f"{k.strip().replace('_', ' ').capitalize()}: {v.strip()}"
                for k, v in self.context.items()
            ]
        else:
            self.retrieve = dspy.Retrieve(k=num_passages)

        self.generate_answer = dspy.ChainOfThought(signature)
        self.kwargs = kwargs

    def forward(self, question):
        # get fixed context if provided
        if self.context is None:
            for hop in range(self.max_hops):
                query = self.generate_query[hop](
                    context=self.context, question=question
                ).query
                passages = self.retrieve(query).passages
                self.context = dspy.deduplicate(self.context + passages)
        else:
            context = self.context


        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)