#!/usr/bin/env python3
"""create_examples.py in src/microchat.

This script generates vector graphics for multiple-choice questions from a CSV file.
"""

import cairo

import pprint as pp
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


def wrap_text(context: cairo.Context, text: str, max_width: int):
    """Wrap text based on the maximum width and the font settings of the context."""

    output_lines = []
    lines = text.split("\n")
    for elem in lines:
        words = elem.split()
        line = ""
        for word in words:
            if word in ["\n"]:
                output_lines.append(line)
                line = ""
                continue

            test_line = f"{line} {word}".strip()
            xbearing, ybearing, width, height, xadvance, yadvance = (
                context.text_extents(test_line)
            )

            if width <= max_width:
                line = test_line
            else:
                output_lines.append(line)
                line = word
        if line:
            output_lines.append(line)

    return output_lines


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
    base_height=200,
    font_size=20,
    section_spacing=30,
    question_prefix: Optional[str] = None,  # "Q"
    explanation_prefix: Optional[str] = "Explanation",
    prediction: Optional[int] = None,
    dry_run=False,
):
    """
    Create a graphic representation of an MCQ with customizable colors and text wrapping.
    """
    # Estimate height based on content
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, base_height)
    context = cairo.Context(surface)
    context.set_font_size(font_size)

    margin_x, margin_y = 50, 40
    y_offset = margin_y

    # Pre-calculate the height needed for the question, options, and explanation
    if question_prefix:
        question = f"{question_prefix}: {question}"

    question_lines = wrap_text(context, f"{question}", width - 2 * margin_x)
    option_lines_wrapped = [
        wrap_text(context, option, width - 2 * margin_x - 10) for option in options
    ]

    # Explanation prefix
    if explanation_prefix:
        explanation = f"{explanation_prefix}: {explanation}"

    # Format explanation text, pprint to use "\n" and "\t" for new lines and tabs
    explanation_lines = pp.pformat(explanation, indent=4, width=width - 2 * margin_x)
    explanation_lines = explanation_lines.split("\n")
    explanation_lines = wrap_text(context, f"{explanation}", width - 2 * margin_x)

    # Dynamic height calculation
    question_height = len(question_lines) * (font_size + line_spacing)
    line_height = font_size - 5  # 10
    options_height = sum(
        len(lines) * (font_size + line_spacing) + line_height
        for lines in option_lines_wrapped
    )
    explanation_height = len(explanation_lines) * (font_size + line_spacing)
    total_height = (
        y_offset + question_height + options_height + explanation_height + margin_y
    )

    # Resize surface if needed
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, int(total_height))
    context = cairo.Context(surface)
    context.set_font_size(font_size)

    # Set up background color
    context.set_source_rgba(*hex_to_rgb(background_color, opacity))
    context.rectangle(0, 0, width, total_height)
    context.fill()

    # Draw Question
    context.set_source_rgba(*hex_to_rgb(question_color, opacity))
    for line in question_lines:
        context.move_to(margin_x, y_offset)
        context.show_text(line)
        y_offset += font_size + line_spacing

    # # Add space between question and options
    # y_offset += section_spacing

    # Draw Options
    pred_color: str = "#000000"
    prediction_coords: dict = None
    for idx, option_lines in enumerate(option_lines_wrapped):
        # Option background and border
        box_height = (len(option_lines) * (font_size + line_spacing)) + line_height
        if idx == correct_index:
            context.set_source_rgba(*hex_to_rgb(answer_background, opacity))
            context.set_line_width(1)
            context.rectangle(margin_x, y_offset, width - 2 * margin_x, box_height)
            context.fill_preserve()
        else:
            context.set_source_rgba(*hex_to_rgb("#FFFFFF", opacity))
            context.set_line_width(1)  # 2*(idx+1))
            context.rectangle(margin_x, y_offset, width - 2 * margin_x, box_height)
            context.fill_preserve()

        if idx == prediction:
            # color box border (all sides)
            pred_color = "#008000" if idx == correct_index else "#FF0000"  # 00FF00
            pred_box_height = (
                len(option_lines) * (font_size + line_spacing)
            ) + line_height
            pred_y_offset = y_offset
            # save prediction coords and draw at the end
            prediction_coords = (
                margin_x,
                pred_y_offset,
                width - 2 * margin_x,
                pred_box_height,
            )
            context.set_source_rgba(*hex_to_rgb(pred_color, opacity))
            context.set_line_width(1)
            context.rectangle(*prediction_coords)
            context.stroke()

        context.set_source_rgb(0, 0, 0)  # Border color
        context.stroke()

        # Option text
        context.set_source_rgba(*hex_to_rgb(option_color, opacity))
        for line in option_lines:
            # I added "+ (line_height)" to center text within the box. It looks okay
            # with font size 20, but may need adjustment to be generalizable.
            context.move_to(margin_x + 10, y_offset + line_height + (line_height))
            context.show_text(line)
            y_offset += font_size + line_spacing
        y_offset += 15

    # Draw prediction border to be on top
    context.set_source_rgba(*hex_to_rgb(pred_color, opacity))
    context.set_line_width(3)
    context.rectangle(*prediction_coords)
    context.stroke()

    # Add space between options and explanation
    y_offset += section_spacing

    # Draw Explanation
    context.set_source_rgba(*hex_to_rgb(explanation_color, opacity))
    for line in explanation_lines:
        context.move_to(margin_x, y_offset)
        context.show_text(line)
        y_offset += font_size + line_spacing

    # Save to PNG
    if not dry_run:
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
    prediction_key: Optional[str] = "pred"
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
        research_subject = " ".join([x[:5] for x in row["research_subject"].split()]) if "research_subject" in row else "mcq"
        research_subject = research_subject.replace(" ", "_").lower()
        use_case = row["use_case"] if "use_case" in row else "na"
        blooms_level = int(row["blooms_level"]) if "blooms_level" in row else "na"

        # get options
        options = eval(row[choices_key])
        options = [str(option).strip() for option in options]
        correct_index = options.index(answer)
        pred_index = row[prediction_key] if prediction_key else None
        pred_correct = "correct" if pred_index == correct_index else "incorrect"

        # get explanation
        explanation = row[explanation_key].strip()

        output_file = f"{idx:05d}_blooms-{blooms_level}_task-{use_case}_{research_subject}_{pred_correct}.png"
        output_path = output_dir.joinpath(output_file)
        create_mcq_graphics(
            question,
            options,
            correct_index,
            explanation,
            output_path,
            prediction=pred_index,
            dry_run=dry_run,
        )

    logger.info("Finished generating graphics.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
