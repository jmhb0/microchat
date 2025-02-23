"""
python -m ipdb analysis_scripts/20241103_create_images_folders.py
"""
import ipdb

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import logging
import os
import pickle
import ast
from PIL import Image
from models.openai_api import call_gpt_batch, call_gpt
import re
from pydantic import BaseModel
from omegaconf import OmegaConf
import logging
from datetime import datetime
import glob
import csv

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from io import BytesIO

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from benchmark.build_raw_dataset.download_data import download_csv

dir_results_parent = Path(__file__).parent / "results" / Path(__file__).stem
dir_results_parent.mkdir(exist_ok=True, parents=True)

key_form=0
key_question_gen = 3 
key_choices_gen = 9
seed=0
f_imgs = Path(f"benchmark/data/formdata_{key_form}/4_images.csv")
df_imgs = pd.read_csv(f_imgs)
cache_images = {}

imgs_all = []
contexts = []
idxs = []

def _get_filenames_from_key(key, ):
    dir_ = f"benchmark/data/formdata_0/images/idx_{key:04d}"
    fs =  [f for f in os.listdir(dir_) if os.path.isfile(os.path.join(dir_, f))]
    fs = [f for f in fs if f!=".DS_Store"]
    fs = sorted(fs)
    return [os.path.join(os.path.join(dir_, f)) for f in fs]


print(f"Collecting images from df_imgs size {len(df_imgs)}")
dir_base = Path("benchmark/data/formdata_0/images")

for key_image, row in df_imgs.iterrows():
    # if key_image > 200:break
    if key_image == 0: 
        continue # was a test image.
    
    fs = _get_filenames_from_key(key_image)

    # try:
    imgs_pil = [Image.open(f).convert('RGB') for f in fs]
    imgs = [np.array(img) for img in imgs_pil]
    imgs_all.append(imgs)

    contexts.append(row['Context - image generation'])

    idxs.append(key_image)

    # except Exception as e:
    #     print(f"/nIssue with files {filenames}")
    #     print(e)
    #     continue

f_save_imgs = dir_results_parent / "images.pkl"
lookup_images = dict(zip(idxs, imgs_all))
with open(f_save_imgs, "wb") as fp:
    pickle.dump(lookup_images, fp)

dir_results_imgs = dir_results_parent / "imgs"
dir_results_imgs.mkdir(exist_ok=True)
dir_results_pdfs = dir_results_parent / "imgs_pdfs"
dir_results_pdfs.mkdir(exist_ok=True)
dir_results_grids = dir_results_parent / "imgs_grids"
dir_results_grids.mkdir(exist_ok=True)

def create_image_pdf(output_path, key_image, images, context_text=None, page_size=letter, margin_inches=0.5, font_size=14, max_height_percent=0.75):
    """
    Creates a PDF containing vertically stacked images and optional context text.
    Images are scaled to 90% of max width or 75% of max height, whichever is smaller, and centered.
    
    Args:
        output_path (str or Path): Path where PDF will be saved
        key_image (int): Image key for title
        images (list): List of numpy arrays
        context_text (str, optional): Text to add at bottom of the PDF
        page_size (tuple): PDF page dimensions (width, height) in points
        margin_inches (float): Margin size in inches
        font_size (int): Font size for context text
        max_height_percent (float): Maximum fraction of page height an image can occupy
    """
    c = canvas.Canvas(str(output_path), pagesize=page_size)
    
    # Calculate dimensions
    page_width, page_height = page_size
    margin = inch * margin_inches
    available_width = page_width - (2 * margin)
    available_height = page_height - (2 * margin)
    max_image_height = available_height * max_height_percent
    
    # Start at top of page
    y_position = page_height - margin
    
    # Add title
    c.setFont("Helvetica", font_size)
    title = f"Image key {key_image}"
    title_width = c.stringWidth(title)
    c.drawString((page_width - title_width) / 2, y_position, title)
    
    # Move position down after title
    y_position -= font_size + margin
    
    # Process each image
    for idx, img_array in enumerate(images):
        if idx < 70 or idx > 90:
            continue
        # Create a temporary file for each image
        temp_path = str(Path(output_path).parent / f"temp_{idx}.png")
        
        # Save numpy array as temporary PNG file
        Image.fromarray(img_array).save(temp_path)
            
        # Get image dimensions
        img = Image.open(temp_path)
        img_width, img_height = img.size
        
        # Calculate scaling factors
        width_scale = (0.9 * available_width) / img_width
        height_scale = max_image_height / img_height  # Use max height limit
        scale_factor = min(width_scale, height_scale)
        
        scaled_width = img_width * scale_factor
        scaled_height = img_height * scale_factor
        
        # Check if we need a new page
        if y_position - scaled_height - margin < margin:
            c.showPage()
            y_position = page_height - margin
            
        # Calculate x position to center image
        x_position = margin + (available_width - scaled_width) / 2
            
        # Draw image from temporary file
        c.drawImage(temp_path, 
                   x_position,
                   y_position - scaled_height,
                   width=scaled_width,
                   height=scaled_height)
        
        # Update position for next image
        y_position -= (scaled_height + margin)
        
        # Clean up temporary file
        Path(temp_path).unlink()
    
    # Add context text if provided
    if context_text:
        text_style = ParagraphStyle(
            'custom',
            fontSize=font_size,
            leading=font_size + 2
        )
        
        # Create paragraph for wrapping
        p = Paragraph(context_text, text_style)
        w, h = p.wrap(available_width, y_position - margin)
        
        # Check if we need a new page for text
        if y_position - h - margin < margin:
            c.showPage()
            y_position = page_height - margin
        
        # Draw text
        p.drawOn(c, margin, y_position - h)
    
    # Save PDF
    c.save()

def create_image_grid(images, target_height=800):
    """
    Create a high-resolution grid of images while preserving aspect ratios.
    For 2 images, places them side by side.
    
    Args:
        images: List of numpy arrays of shape (H,W,3) with potentially different dimensions
        target_height: Target height for each image in pixels (default: 800)
        
    Returns:
        grid: Numpy array containing the arranged images in a grid
    """
    import numpy as np
    from math import ceil, sqrt
    import cv2
    
    # Determine number of images
    n_images = len(images)
    if n_images == 0:
        return None
    
    # Resize images to have the same height while preserving aspect ratio
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio)
        # Use INTER_LANCZOS4 for high-quality downscaling
        resized = cv2.resize(img, (new_width, target_height), 
                           interpolation=cv2.INTER_LANCZOS4)
        resized_images.append(resized)
    
    # Special handling for 2 images - put them side by side
    if n_images == 2:
        total_width = sum(img.shape[1] for img in resized_images)
        grid = np.zeros((target_height, total_width, 3), dtype=np.uint8)
        
        # Place first image
        current_x = 0
        grid[:, current_x:current_x + resized_images[0].shape[1]] = resized_images[0]
        
        # Place second image
        current_x += resized_images[0].shape[1]
        grid[:, current_x:current_x + resized_images[1].shape[1]] = resized_images[1]
        
        return grid
    
    # For other numbers of images, use the square grid approach
    grid_size = ceil(sqrt(n_images))
    grid_height = grid_size
    grid_width = grid_size
    
    # Find maximum width among resized images
    max_width = max(img.shape[1] for img in resized_images)
    
    # Create empty grid with padding between images
    padding = 20  # Pixels of padding between images
    grid_total_height = grid_height * target_height + (grid_height - 1) * padding
    grid_total_width = grid_width * max_width + (grid_width - 1) * padding
    grid = np.zeros((grid_total_height, grid_total_width, 3), dtype=np.uint8)
    
    # Place images in grid
    for idx, img in enumerate(resized_images):
        i = idx // grid_width
        j = idx % grid_width
        
        # Calculate centering offset for images narrower than max_width
        width_offset = (max_width - img.shape[1]) // 2
        
        # Calculate positions including padding
        y_start = i * (target_height + padding)
        y_end = y_start + target_height
        x_start = j * (max_width + padding) + width_offset
        x_end = x_start + img.shape[1]
        
        grid[y_start:y_end, x_start:x_end] = img
    
    return grid

pickle_only = True
grids = []
for key_image, imgs, context in zip(idxs, imgs_all, contexts):
    print("key image", key_image)
    if key_image <= 227:  # Skip already processed keys
        continue
    
    # save images one-by-one
    for i, img in enumerate(imgs): 
        fname = dir_results_imgs / f"img_{key_image:03d}_{i}.png"
        Image.fromarray(img).save(fname)

    # save pdfs 
    fname_pdf = dir_results_pdfs / f"img_{key_image:03d}.pdf"
    create_image_pdf(fname_pdf, key_image, imgs, context)

    # save grids if there are multiple images per submission
    grid = create_image_grid(imgs)
    fname = dir_results_grids / f"grid_{key_image:03d}.png"
    Image.fromarray(grid).save(fname)
    # ipdb.set_trace()

    # grids.append(grid)

# now make a super-grid 
# supergrid = create_image_grid(grids)
Image.fromarray(supergrid).save(dir_results_parent / "supergrid.png")

ipdb.set_trace()
pass

