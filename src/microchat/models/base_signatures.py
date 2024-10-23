#!/usr/bin/env python3
"""base_signatures.py in src/microchat/models."""

import dspy


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
