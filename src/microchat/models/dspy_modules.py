#!/usr/bin/env python3
"""dspy_modules.py in src/microchat/models."""

from pathlib import Path
from typing import List, Optional

from loguru import logger

import dspy


from microchat import MODULE_ROOT

from microchat.fileio.text.readers import yaml_loader
from microchat.models.base_signatures import ReviseQuestion, ReviseQuestionContext, ClassifyBlooms, GenerateSearchQuery

context = yaml_loader(Path(MODULE_ROOT, "conf", "question_context.yaml"))

# Base class for RAG Modules
class BaseRAG(dspy.Module):
    def __init__(self, num_passages: int = 5, **kwargs):
        """Initialize shared components for RAG modules."""
        super().__init__()
        self.num_passages = num_passages
        self.context = None
        self.retrieve = None
        self.signature: Optional[dspy.Signature] = None
        self.kwargs = kwargs
        self._set_context_and_signature()

    def _set_context_and_signature(self):
        """Set context and signature based on specified context type."""
        if self.kwargs.get("context") == "nbme":
            self.signature = ReviseQuestionContext
            self.context = self._format_context(context["nbme"])
        elif self.kwargs.get("context") == "blooms":
            self.signature = ClassifyBlooms
            self.context = self._format_context(context["blooms"])
        else:
            self.retrieve = dspy.Retrieve(k=self.num_passages)


    @staticmethod
    def _format_context(raw_context: dict) -> List[str]:
        """Format context into list of strings with capitalized keys and stripped values."""
        return [
            f"{k.strip().replace('_', ' ').capitalize()}: {v.strip()}"
            for k, v in raw_context.items()
        ]

    def generate_answer(self, question: str, context: Optional[List[str]] = None):
        """Generate an answer using the given context and question."""
        if context is None:
            context = self.retrieve(question).passages if self.retrieve else self.context
        prediction = dspy.ChainOfThought(self.signature)(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


# Define the module
# Specific RAG module implementations
class CoTRAG(BaseRAG):
    def __init__(self, num_passages: int = 5, **kwargs):
        """Initialize the CoTRAG module with specified context and passages."""
        super().__init__(num_passages=num_passages, **kwargs)
        if not self.signature:
            self.signature = ReviseQuestion  # Default signature if none is specified

    def forward(self, question: str):
        """Forward method for processing the question through the RAG pipeline."""
        return self.generate_answer(question)


class CoTMultiHopRAG(BaseRAG):
    """Module for multi-hop reasoning with multiple query hops."""
    def __init__(self, num_passages: int = 5, passages_per_hop: int = 3, max_hops: int = 3, **kwargs):
        super().__init__(num_passages=num_passages, **kwargs)
        self.passages_per_hop = passages_per_hop
        self.max_hops = max_hops
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        if not self.signature:
            self.signature = ReviseQuestion  # Default signature if none is specified

    def forward(self, question: str):
        """Multi-hop forward method for iterative retrieval and answering."""
        if self.context is None:
            for hop in range(self.max_hops):
                query = self.generate_query[hop](context=self.context, question=question).query
                passages = self.retrieve(query).passages
                self.context = dspy.deduplicate(self.context + passages)
        return self.generate_answer(question)