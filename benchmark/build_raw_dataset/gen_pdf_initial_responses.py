"""
Put each form response into a pdf. 
Different from gen_pdf_responses.py, it does not include the GPT-generated distractors or GPT responses.

python benchmark/build_raw_dataset/gen_pdf_initial_responses.py
"""
import ipdb
import pandas as pd
import os
import tqdm
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage
from aicsimageio import AICSImage

def get_styles():
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    bold_style = ParagraphStyle('Bold', parent=styles['Normal'], fontName='Helvetica-Bold')
    header_style = ParagraphStyle('Header', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
    return normal_style, bold_style, header_style

def truncate_text(text, max_length=200):
    return (text[:max_length] + '...') if len(text) > max_length else text

def resize_image(img_path, max_width, max_height):
    with PILImage.open(img_path) as img:
        img_width, img_height = img.size
        aspect = img_height / float(img_width)
        
        if img_width > max_width:
            new_width = max_width
            new_height = new_width * aspect
        else:
            new_width = img_width
            new_height = img_height
        
        if new_height > max_height:
            new_height = max_height
            new_width = new_height / aspect
        
        return new_width, new_height

def create_pdf(idx, row, output_dir, formdata_dir, normal_style, bold_style, header_style):
    submitter_name = row.get("Your name", "Unknown")
    safe_submitter_name = ''.join(c if c.isalnum() else '_' for c in submitter_name)
    pdf_filename = os.path.join(output_dir, f'response_{idx}_{safe_submitter_name}.pdf')

    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, leftMargin=inch, rightMargin=inch)
    story = []

    page_width, page_height = letter
    max_img_height = page_height / 4
    available_width = page_width - 2*inch

    for col, value in row.items():
        if col == 'Image / image set':  
            img_dir = os.path.join(formdata_dir, "images", f'idx_{idx:04d}')
            if os.path.exists(img_dir):
                for fname in os.listdir(img_dir):
                    file_path = os.path.join(img_dir, fname)
                    if fname.lower().endswith(('.tif', '.tiff')):
                        try:
                            img = AICSImage(file_path)
                            story.append(Paragraph(f"TIFF file: {fname}", bold_style))
                            story.append(Paragraph(f"Image shape: {img.shape}", normal_style))
                            story.append(Paragraph(f"Image dimensions: {img.dims}", normal_style))
                            story.append(Spacer(1, 12))
                            
                            try:
                                with PILImage.open(file_path) as pil_img:
                                    preview_width, preview_height = resize_image(file_path, available_width, max_img_height)
                                    temp_preview_path = os.path.join(output_dir, f"temp_preview_{fname}.png")
                                    pil_img.save(temp_preview_path, "PNG")
                                    story.append(Image(temp_preview_path, width=preview_width, height=preview_height))
                                    story.append(Spacer(1, 12))
                                    os.remove(temp_preview_path)
                            except Exception as preview_error:
                                story.append(Paragraph(f"Could not generate preview for {fname}: {str(preview_error)}", normal_style))
                                story.append(Spacer(1, 12))
                        except Exception as e:
                            story.append(Paragraph(f"Error processing TIFF {fname}: {str(e)}", normal_style))
                            story.append(Spacer(1, 12))
                    elif fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        try:
                            image_width, image_height = resize_image(file_path, available_width, max_img_height)
                            story.append(Image(file_path, width=image_width, height=image_height))
                            story.append(Spacer(1, 12))
                        except Exception as e:
                            story.append(Paragraph(f"Error loading {fname}: {str(e)}", normal_style))
                            story.append(Spacer(1, 12))
                    else:
                        story.append(Paragraph(f"{fname} not loaded (unsupported format)", normal_style))
                        story.append(Spacer(1, 12))
        else:
            truncated_col = truncate_text(col)
            story.append(Paragraph(truncated_col, bold_style))
            story.append(Spacer(1, 6))
            story.append(Paragraph(str(value), normal_style))
            story.append(Spacer(1, 12))

    doc.build(story)

import pandas as pd
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage
from aicsimageio import AICSImage

def filter_columns(df):
    # Remove columns that start with "update", "bak", or "iloc"
    return df[[col for col in df.columns if not col.startswith(('update', 'bak', 'iloc'))]]

def reorder_columns(df):
    columns = df.columns.tolist()
    new_order = []
    question_related_columns = {}

    # First, group all question-related columns
    for col in columns:
        if col.startswith('Question ') and 'use case' not in col:
            question_number = col.split()[1]
            if question_number not in question_related_columns:
                question_related_columns[question_number] = {'Question': col}
        elif col.startswith('Answer '):
            question_number = col.split()[1]
            if question_number in question_related_columns:
                question_related_columns[question_number]['Answer'] = col
        elif col.startswith('Comments about question '):
            question_number = col.split()[-1]
            if question_number in question_related_columns:
                question_related_columns[question_number]['Comments'] = col
        elif col.startswith('Incorrect answer '):
            question_number = col.split()[-1]
            if question_number in question_related_columns:
                question_related_columns[question_number]['Incorrect'] = col
        elif 'use case' in col:
            question_number = col.split()[1]
            if question_number in question_related_columns:
                question_related_columns[question_number]['Use Case'] = col

    # Now, create the new order
    for col in columns:
        if not any(col.startswith(prefix) for prefix in ('Question ', 'Answer ', 'Comments about question ', 'Incorrect answer ')) and 'use case' not in col:
            new_order.append(col)

    # Custom sorting function to handle both numeric and non-numeric keys
    def sort_key(x):
        try:
            return int(x)
        except ValueError:
            return float('inf')  # Place non-numeric keys at the end

    # Add question-related columns in the desired order
    for question_number in sorted(question_related_columns.keys(), key=sort_key):
        cols = question_related_columns[question_number]
        new_order.extend([
            cols.get('Question', ''),
            cols.get('Use Case', ''),
            cols.get('Answer', ''),
            cols.get('Comments', ''),
            cols.get('Incorrect', '')
        ])

    # Remove empty strings and duplicates while preserving order
    new_order = list(dict.fromkeys([col for col in new_order if col]))

    return df[new_order]

def process_responses(idx_form):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    formdata_dir = os.path.join("benchmark/data", f'formdata_{idx_form}')
    csv_path = os.path.join(formdata_dir, '1_responses_after_edits0.csv')
    output_dir = os.path.join(formdata_dir, 'form_responses')
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path, dtype=str)
    df = df.map(lambda x: '' if pd.isna(x) else str(x))
    df = filter_columns(df)  # Remove unwanted columns
    df = reorder_columns(df)  # Reorder the remaining columns
    normal_style, bold_style, header_style = get_styles()

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        try:
            create_pdf(idx, row, output_dir, formdata_dir, normal_style, bold_style, header_style)
        except Exception as e:
            print(f"Error creating PDF for row {idx}: {str(e)}")

    print(f"PDFs have been created in the '{output_dir}' directory.")



def main():
    idx_form = 0  # Set this to your desired form index
    process_responses(idx_form)

if __name__ == "__main__":
    main()