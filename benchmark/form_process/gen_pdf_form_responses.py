import pandas as pd
import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage
from aicsimageio import AICSImage
import ipdb

def get_styles():
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    bold_style = ParagraphStyle('Bold', parent=styles['Normal'], fontName='Helvetica-Bold')
    return normal_style, bold_style

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

def create_pdf(idx, row, output_dir, formdata_dir, normal_style, bold_style, qas=None):
    pdf_filename = os.path.join(output_dir, f'response_{idx}.pdf')
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, leftMargin=inch, rightMargin=inch)
    story = []

    page_width, page_height = letter
    max_img_height = page_height / 4
    available_width = page_width - 2*inch

    for i, (col, value) in enumerate(row.items()):
        if i == 3:  # Fourth column
            img_dir = os.path.join(formdata_dir, "images", f'idx_{idx}')
            if os.path.exists(img_dir):
                for fname in os.listdir(img_dir):
                    file_path = os.path.join(img_dir, fname)
                    if fname.lower().endswith(('.tif', '.tiff')):
                        try:
                            img = AICSImage(file_path)
                            story.append(Paragraph(f"TIFF file: {fname}", bold_style))
                            story.append(Paragraph(f"Image shape: {img.shape}", normal_style))
                            story.append(Paragraph("", normal_style))  # Line break
                            story.append(Paragraph(f"Image dimensions: {img.dims}", normal_style))
                            story.append(Spacer(1, 12))
                            
                            # TIFF preview (if possible)
                            try:
                                with PILImage.open(file_path) as pil_img:
                                    preview_width, preview_height = resize_image(file_path, available_width, max_img_height)
                                    temp_preview_path = os.path.join(output_dir, f"temp_preview_{fname}.png")
                                    pil_img.save(temp_preview_path, "PNG")
                                    story.append(Image(temp_preview_path, width=preview_width, height=preview_height))
                                    story.append(Spacer(1, 12))
                                    os.remove(temp_preview_path)  # Clean up the temporary file
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

    # Add generated questions and choices if available
    if qas and str(idx) in qas:
        story.append(Paragraph("Generated Questions:", bold_style))
        story.append(Spacer(1, 12))
        for i, qa in enumerate(qas[str(idx)], 1):
            story.append(Paragraph(f"Question {i}: {qa['question']}", normal_style))
            story.append(Spacer(1, 6))
            for j, choice in enumerate(qa['choices'], 0):
                story.append(Paragraph(f"{j}: {choice}", normal_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Answer: {qa['answer']}", normal_style))
            story.append(Spacer(1, 12))

    doc.build(story)

def process_responses(idx_form, show_gen_questions=False, prompt_key=0, seed=0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    formdata_dir = os.path.join(script_dir, f'formdata_{idx_form}')
    csv_path = os.path.join(formdata_dir, 'responses.csv')
    
    if show_gen_questions:
        output_dir = os.path.join(formdata_dir, 'form_responses_w_choices', f'keyprompt_{prompt_key}_seed_{seed}')
    else:
        output_dir = os.path.join(formdata_dir, 'form_responses')
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    normal_style, bold_style = get_styles()

    qas = None

    if show_gen_questions:
        qa_file = os.path.join(formdata_dir, 'generated_questions_text', f'qa_keyprompt_{prompt_key}_seed_{seed}.json')
        if os.path.exists(qa_file):
            with open(qa_file, 'r') as f:
                qas = json.load(f)

    for idx, row in df.iterrows():
        try:
            create_pdf(idx, row, output_dir, formdata_dir, normal_style, bold_style, qas)
        except Exception as e:
            print(f"Error creating PDF for row {idx}: {str(e)}")

    print(f"PDFs have been created in the '{output_dir}' directory.")

def main():
    idx_form = 0  # Change this value as needed
    show_gen_questions = True  # Set to False to use the original behavior
    prompt_key = 0  # Default value
    seed = 0  # Default value
    process_responses(idx_form, show_gen_questions, prompt_key, seed)

if __name__ == "__main__":
    main()
