"""
Must be run after eval.py for the same parameters
python eval_as_pdf.py
"""
import ipdb
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image as PILImage
import io

def format_text(text):
    """Replace newline characters with HTML line breaks."""
    return str(text).replace('\n', '<br/>')

def make_pdf(key_form, key_question_gen, key_choices_gen, seed, model):
    dir_pdfs = Path(
        f"benchmark/data/formdata_{key_form}/question_strategy_{key_question_gen}/pdfs_questions_key_choices_{key_choices_gen}_evalclosed_{model}"
    )
    dir_pdfs.mkdir(exist_ok=True, parents=True)
    
    f_eval_closed = Path(
        f"benchmark/data/formdata_{key_form}/question_strategy_{key_question_gen}/df_questions_key_choices_{key_choices_gen}_evalclosed_{model}.csv"
    )
    df_questions = pd.read_csv(f_eval_closed, index_col='key_question')
    ipdb.set_trace()
    f_images = f"benchmark/data/formdata_{key_form}/4_images.csv"
    df_images = pd.read_csv(f_images, index_col='key_image')
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Normal_LEFT', 
                              parent=styles['Normal'],
                              alignment=0,  # 0 is left alignment
                              spaceAfter=6))

    for idx, row in df_images.iterrows():
        pdf_path = dir_pdfs / f"idx_{idx}.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        story = []

        # Load and add images
        fnames_images = ast.literal_eval(row['fnames_images'])
        skip_pdf = False
        for fname in fnames_images:
            img_path = Path(row['dir_imgs']) / fname
            try:
                img = PILImage.open(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                skip_pdf = True
                break

            # Calculate aspect ratio
            aspect = img.width / img.height
            
            # Set maximum width and height
            max_width = 6 * inch
            max_height = 4 * inch
            
            # Calculate new dimensions
            if aspect > max_width / max_height:  # wider than tall
                new_width = max_width
                new_height = new_width / aspect
            else:  # taller than wide
                new_height = max_height
                new_width = new_height * aspect
            
            img_data = io.BytesIO()
            img.save(img_data, format='PNG')
            img_data.seek(0)
            img = Image(img_data, width=new_width, height=new_height)
            story.append(img)
            story.append(Spacer(1, 12))

        if skip_pdf:
            print(f"Skipping PDF for idx_{idx} due to image loading error")
            continue

        # Add text information
        cols_to_display = ['Images - source 1', 'Images source 2', 'Context - image generation', 'Context - motivation', 'caption', 'key_person']
        for col in cols_to_display:
            story.append(Paragraph(f"<b>{col}:</b> {format_text(row[col])}", styles['Normal_LEFT']))

        # Add questions and choices
        questions = df_questions[df_questions['key_image'] == idx]
        if len(questions) == 0: 
            continue

        questions = questions.sort_values('question_number')
        for _, q_row in questions.iterrows():
            story.append(Paragraph(f"<b>Question {q_row['question_number']}:</b> {format_text(q_row['question'])}", styles['Normal_LEFT']))
            
            # Parse the choices dict using ast.literal_eval
            try:
                choices_dict = ast.literal_eval(q_row['choices'])
            except:
                print(f"Error parsing choices for question {q_row['question_number']}")
                print(f"Choices string: {q_row['choices']}")
                choices_dict = {'choices': ['Error parsing choices'], 'correct_index': -1}
            
            choices = choices_dict['choices']
            correct_index = choices_dict['correct_index'] + 1
            
            # Format choices
            formatted_choices = [f"{i+1}. {format_text(choice)}" for i, choice in enumerate(choices)]
            formatted_choices.append(f"Correct index: {correct_index}")
            
            choices_text = "<br/>".join(formatted_choices)
            story.append(Paragraph(f"<b>Choices {q_row['question_number']}:</b><br/>{choices_text}", styles['Normal_LEFT']))

            # Add Original Answer
            story.append(Paragraph(f"<b>Original Answer {q_row['question_number']}:</b> {format_text(q_row['answer'])}", styles['Normal_LEFT']))

            # Add GPT prediction and response
            gpt_prediction = int(q_row['gpt_prediction']) + 1
            story.append(Paragraph(f"<b>GPT Prediction:</b> {format_text(gpt_prediction)}", styles['Normal_LEFT']))
            story.append(Paragraph(f"<b>GPT Response:</b> {format_text(q_row['gpt_response'])}", styles['Normal_LEFT']))
            story.append(Spacer(1, 12))

        doc.build(story)

    print(f"PDFs generated in {dir_pdfs}")

if __name__ == "__main__":
    # which form we collect the questions from
    key_form = 0
    # which set of questions to get - made in make_questions.py
    key_question_gen = 0
    # key for generating the choices
    key_choices_gen = 0
    model = 'gpt-4o-mini'
    make_pdf(key_form, key_question_gen, key_choices_gen, seed=0, model=model)
    ipdb.set_trace()
    pass