#!/usr/bin/env python3
"""create_examples.py in src/microchat.

This script generates vector graphics for multiple-choice questions from a CSV file.
"""

import pandas as pd
import cairo
from PIL import Image
from pathlib import Path
import click
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger

from microchat import LOG_DIR


def hex_to_rgb(hex_color: str, opacity: float = 1.0) -> tuple:
    """Convert hex color to an RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return (*rgb, opacity)


def wrap_text(context, text, max_width):
    """Wrap text based on the maximum width and the font settings of the context."""
    words = text.split()
    lines = []
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        xbearing, ybearing, width, height, xadvance, yadvance = context.text_extents(
            test_line
        )

        if width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    return lines


def create_mcq_graphics(
    question: str,
    options: list,
    correct_index: int,
    explanation: str,
    output_path: Path,
    question_color="#000000",
    option_color="#000000",
    background_color="#FFFFFF",
    answer_background="#DFFFD6",
    explanation_color="#333333",
    opacity=1.0,
    line_spacing=10,
    width=800,
    height=600,
    font_size=20,
    dry_run=False,
):
    """
    Create a graphic representation of an MCQ with customizable colors and text wrapping.

    Args:
        question (str): The question text.
        answer (str): The correct answer.
        options (list): List of options.
        explanation (str): Explanation text.
        output_path (Path): Path to save the generated graphic.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)

    # Set up background color
    context.set_source_rgba(*hex_to_rgb(background_color, opacity))
    context.rectangle(0, 0, width, height)
    context.fill()

    # Fonts and colors
    context.set_font_size(font_size)
    margin_x, margin_y = 50, 40

    # Draw Question
    context.set_source_rgba(*hex_to_rgb(question_color, opacity))
    y_offset = margin_y
    lines = wrap_text(context, f"Q: {question}", width - 2 * margin_x)
    for line in lines:
        context.move_to(margin_x, y_offset)
        context.show_text(line)
        y_offset += 15 + line_spacing

    # Draw Options
    if correct_index not in range(len(options)):
        raise ValueError("Correct index must be within the range of options.")

    #
    for idx, option in enumerate(options):
        box_height = 30
        context.rectangle(margin_x, y_offset, width - 2 * margin_x, box_height)

        # Option text
        if idx == correct_index:
            context.rectangle(margin_x, y_offset, width - 2 * margin_x, box_height)
            context.set_source_rgba(
                *hex_to_rgb(answer_background, opacity)  # answer_color
            )  # Background for answer
            context.fill_preserve()  # fill the rectangle
            context.set_line_width(2)  # thick border for correct answer
            context.set_source_rgb(0, 0, 0)  # Border color
            context.stroke()  # draw border
        else:
            # Background for options
            context.set_source_rgba(*hex_to_rgb("#FFFFFF", opacity))
            context.fill_preserve()
            context.set_source_rgb(0, 0, 0)  # Border color
            context.stroke()
            context.set_source_rgba(*hex_to_rgb(option_color, opacity))

        option_lines = wrap_text(context, option, width - 2 * margin_x - 10)
        for opt_line in option_lines:
            context.move_to(margin_x + 10, y_offset + 20)
            context.show_text(opt_line)
            y_offset += 20
        y_offset += 15

    # Draw Explanation
    context.set_source_rgba(*hex_to_rgb(explanation_color, opacity))
    explanation_lines = wrap_text(
        context, f"Explanation: {explanation}", width - 2 * margin_x
    )
    for line in explanation_lines:
        context.move_to(margin_x, y_offset)
        context.show_text(line)
        y_offset += 25

    # Save to PNG and convert to image
    if dry_run:
        logger.info("Dry run: no changes will be made.")
        return

    surface.write_to_png(output_path)


@click.command()
@click.argument(
    "input-file", type=click.Path(dir_okay=False, exists=True, path_type=Path)
)
@click.option(
    "--output-dir", type=click.Path(file_okay=False, exists=False, path_type=Path)
)
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    input_file: Path,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    """Generate MCQ graphics from a CSV file."""
    output_dir = output_dir or input_file.parent.joinpath(input_file.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        LOG_DIR.joinpath(f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")

    if input_file.suffix == ".csv":
        df = pd.read_csv(input_file)
    elif input_file.suffix == ".xlsx":
        df = pd.read_excel(input_file)
    else:
        raise ValueError("Input file must be a CSV or Excel file.")

    logger.info(f"Loaded {len(df)} rows from input file.")
    question_key = "question_2"
    answer_key = "answer_2_formatted"
    choices_key = "choices_2"
    explanation_key = "msg"
    metadata_cols = [
        "blooms_level",
        "use_case",
        "organism",
        "specimen",
        "research_subject",
    ]
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row[question_key]
        answer = row[answer_key].strip()

        # get options
        options = eval(row[choices_key])
        options = [str(option).strip() for option in options]
        correct_index = options.index(answer)
        explanation = row[explanation_key].strip()

        output_path = output_dir.joinpath(f"mcq_{idx}.png")
        create_mcq_graphics(
            question,
            answer,
            options,
            correct_index,
            explanation,
            output_path,
            dry_run=dry_run,
        )

    logger.info("Finished generating graphics.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
