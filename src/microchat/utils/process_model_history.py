#!/usr/bin/env python3
"""process_model_history.py in src/microchat/utils."""


import json
import pprint
from pathlib import Path
from typing import Union, Optional

from loguru import logger


def history_to_jsonl(
    lm, output_dir: Union[str, Path], n: int = 1, output_file: Optional[str] = None
):
    """
    Writes the last n prompts and their completions to a JSONL file.

    Args:
        lm: The language model with a `history` attribute.
        output_dir: The directory to write the JSONL file to.
        n: The number of recent history items to include.
        output_file: Path to the JSONL file to write.
    """
    output_file = output_file or f"{lm.model_name}_history.jsonl"
    output_path = Path(output_dir).joinpath(output_file)

    output_prompt_dict = {}
    with output_path.open("w") as f:
        for item in lm.history[-n:]:
            entry = {
                "timestamp": item.get("timestamp", "Unknown time"),
                "messages": item.get("messages")
                or [{"role": "user", "content": item["prompt"]}],
                "responses": item.get("outputs", []),
            }
            f.write(json.dumps(entry) + "\n")

            # save system_context, and user message to json
            system_context = None
            user_message = None
            if message_list := item.get("messages"):
                # find the message with role 'system'
                for message in message_list:
                    if message.get("role") == "system":
                        system_context = message.get("content")

                    elif message.get("role") == "user":
                        user_message = message.get("content")
                    else:
                        logger.warning(f"Unknown message role: {message.get('role')}")

                # update the output_prompt_dict
                output_prompt_dict[item["uuid"]] = {
                    "timestamp": entry["timestamp"],
                    "system_context": system_context,
                    "user_message": user_message,
                }

    logger.info(f"History written to {output_path}")
    # save json file with optimized prompt, system_context, and user_message
    output_file = output_path.parent.joinpath(f"{lm.model_name}_dspy_prompt.json")
    logger.info(f"Writing optimized prompts to {output_file}")
    with open(output_file, "w") as f:
        json.dump(output_prompt_dict, f, indent=4)
