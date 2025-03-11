"""
python -m ipdb analysis_scripts/20250225_fix_imgs_73_74_75.py
"""

import os
import ipdb
import sys
import json
from pathlib import Path
import numpy as np
from collections import Counter
from PIL import Image
import tifffile

sys.path.insert(0, "..")
sys.path.insert(0, ".")

dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)
print(dir_results)

# Create grids directory
grid_dir = dir_results / "grids"
grid_dir.mkdir(exist_ok=True, parents=True)
print(f"Created grid directory: {grid_dir}")

# Create image_updates directory
updates_dir = dir_results / "image_updates"
updates_dir.mkdir(exist_ok=True, parents=True)
print(f"Created image updates directory: {updates_dir}")

for idx in (73, 75, 75):
    dir_idx = dir_results / f"idx_{idx:d}"
    print(f"Checking folder: {dir_idx}")
    
    # Assert folder exists
    assert dir_idx.exists(), f"Directory {dir_idx} does not exist"
    
    # Create subfolder for this idx in grids directory
    idx_grid_dir = grid_dir / f"idx_{idx:d}"
    idx_grid_dir.mkdir(exist_ok=True, parents=True)
    print(f"Created grid subfolder: {idx_grid_dir}")
    
    # Create subfolder for this idx
    idx_updates_dir = updates_dir / f"idx_{idx:d}"
    idx_updates_dir.mkdir(exist_ok=True, parents=True)
    print(f"Created updates subfolder: {idx_updates_dir}")
    
    # Get all tif/tiff files
    tiff_files = list(dir_idx.glob("*.tif*"))
    print(f"Found {len(tiff_files)} tif/tiff files in {dir_idx}:")
    for tiff_file in tiff_files:
        print(f"Processing {tiff_file.name}")
        with tifffile.TiffFile(tiff_file) as tif:
            img = tif.asarray()
            print(f"    TIFF shape: {img.shape}")
            
            # Get max projections for each channel
            channel0 = img[0]  # Shape: (6,h,w)
            channel1 = img[1]  # Shape: (6,h,w)
            
            max_ch0 = np.max(channel0, axis=0)  # Shape: (h,w)
            max_ch1 = np.max(channel1, axis=0)  # Shape: (h,w)
            
            # Normalize to [0,1]
            norm_ch0 = (max_ch0 - max_ch0.min()) / (max_ch0.max() - max_ch0.min() + 1e-8)
            norm_ch1 = (max_ch1 - max_ch1.min()) / (max_ch1.max() - max_ch1.min() + 1e-8)
            
            # Create RGB composite
            composite = np.zeros((max_ch0.shape[0], max_ch0.shape[1], 3))
            composite[..., 0] = norm_ch0  # Red channel
            composite[..., 1] = norm_ch1  # Green channel
            
            # Convert to uint8 for saving
            ch0_uint8 = (norm_ch0 * 255).astype(np.uint8)
            ch1_uint8 = (norm_ch1 * 255).astype(np.uint8)
            composite_uint8 = (composite * 255).astype(np.uint8)
            
            # Stack horizontally
            h, w = ch0_uint8.shape
            final_img = np.zeros((h, w * 3, 3), dtype=np.uint8)
            
            # Place grayscale images (repeated across RGB channels)
            final_img[:, 0:w, :] = ch0_uint8[..., None].repeat(3, axis=2)
            final_img[:, w:2*w, :] = ch1_uint8[..., None].repeat(3, axis=2)
            final_img[:, 2*w:3*w, :] = composite_uint8
            
            # Save as PNG
            output_path = idx_updates_dir / f"{tiff_file.stem}_composite.png"
            Image.fromarray(final_img).save(output_path)
            print(f"    Saved composite image to {output_path}")
            
            # Print metadata if available
            if hasattr(tif, 'imagej_metadata'):
                print(f"    ImageJ metadata: {tif.imagej_metadata}")
            if hasattr(tif, 'metadata'):
                print(f"    TIFF metadata: {tif.metadata}")
    ipdb.set_trace()
    print()
