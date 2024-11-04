#!/usr/bin/env python3
"""process_text.py in src/microchat/utils."""
from pathlib import Path
from typing import Tuple


import re
from microchat import MODULE_ROOT
from typing import Optional
from loguru import logger


import tiktoken as tk
from microchat.fileio.text.readers import yaml_loader


context = yaml_loader(Path(MODULE_ROOT, "conf", "question_context.yaml"))
blooms_dict = yaml_loader(Path(MODULE_ROOT, "conf", "blooms.yaml"))["taxonomy"].get(
    "revised"
)
blooms_list = [item for sublist in blooms_dict.values() for item in sublist]
re_blooms_compiled = re.compile(r"|".join(blooms_list), re.IGNORECASE)


def process_blooms(
    answer: str, reference_dict: Optional[dict] = blooms_dict
) -> Tuple[int, str]:
    # extract the blooms level from the response
    blooms_name = None
    blooms_level = -1
    if match := re_blooms_compiled.search(answer):
        blooms_name = match.group().lower()
        # find the level of the blooms taxonomy from blooms_dict
        blooms_level = next(
            level for level, names in reference_dict.items() if blooms_name in names
        )
    else:
        logger.warning(f"Bloom's taxonomy level found in answer: {answer}")

    return blooms_level, blooms_name


def compute_chars(prompt: str) -> int:
    """Compute the number of characters in a prompt."""
    return len(prompt)


def compute_tokens(prompt: str, tokenizer: tk.Encoding) -> int:
    """Compute the number of tokens in a prompt."""
    return len(tokenizer.encode(prompt))
