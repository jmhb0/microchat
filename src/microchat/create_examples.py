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
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (*rgb, opacity)


def wrap_text(context, text, max_width):
    """Wrap text based on the maximum width and the font settings of the context."""
    words = text.split()
    lines = []
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        xbearing, ybearing, width, height, xadvance, yadvance = context.text_extents(test_line)

        if width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    return lines

def create_mcq_graphics(question: str, answer: str, options: list, explanation: str, output_path: Path, dry_run: bool = False) -> None:
    """
    Create a graphic representation of an MCQ using cairo and Pillow.

    Args:
        question (str): The question text.
        answer (str): The correct answer.
        options (list): List of options.
        explanation (str): Explanation text.
        output_path (Path): Path to save the generated graphic.
    """
    width, height = 800, 600
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)

    # Background color
    context.set_source_rgb(1, 1, 1)
    context.rectangle(0, 0, width, height)
    context.fill()

    # Fonts and colors
    context.set_source_rgb(0, 0, 0)
    context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    context.set_font_size(20)

    # Draw question
    context.move_to(50, 50)
    context.show_text(f"Q: {question}")

    # Draw options with colored boxes
    box_y = 100
    for option in options:
        context.rectangle(40, box_y, 720, 30)
        context.set_source_rgb(0.9, 0.9, 0.9)  # Light gray background for options
        context.fill_preserve()
        context.set_source_rgb(0, 0, 0)
        context.stroke()
        context.move_to(50, box_y + 20)
        context.show_text(option)
        box_y += 40

    # Draw answer box
    context.rectangle(40, box_y, 720, 30)
    context.set_source_rgb(0.8, 1.0, 0.8)  # Light green background for correct answer
    context.fill_preserve()
    context.set_source_rgb(0, 0, 0)
    context.stroke()
    context.move_to(50, box_y + 20)
    context.show_text(f"Answer: {answer}")

    # Draw explanation
    context.move_to(50, box_y + 60)
    context.show_text(f"Explanation: {explanation}")

    # Save to PNG and convert to image
    if dry_run:
        logger.info("Dry run: no changes will be made.")
        return

    surface.write_to_png(output_path)
    Image.open(output_path).show()  # Display image for reference


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
    output_dir =  output_dir or input_file.parent.joinpath(input_file.stem)
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
    metadata_cols = ["blooms_level", "use_case", "organism", "specimen", "research_subject"]
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row[question_key]
        answer = row[answer_key]

        # get options
        options = eval(row[choices_key])
        options = [str(option).strip() for option in options]
        explanation = row[explanation_key]

        output_path = output_dir.joinpath(f"mcq_{idx}.png")
        create_mcq_graphics(question, answer, options, explanation, output_path, dry_run=dry_run)

    logger.info("Finished generating graphics.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
