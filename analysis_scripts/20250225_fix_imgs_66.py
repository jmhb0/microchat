"""
python -m ipdb analysis_scripts/20250225_fix_imgs_66.py
"""

import os
import ipdb
import sys
import json
from pathlib import Path
import tifffile
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
sys.path.insert(0, ".")

dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)
print(dir_results)

# Load the TIFF file
dir_idx_66 = Path("analysis_scripts/results/20250225_fix_imgs_66/idx_66")
tiff_file = next(dir_idx_66.glob("*.tif*"))  # Get the first TIFF file in the directory
img = tifffile.imread(tiff_file)
print(f"Image shape: {img.shape}")

# Check the shape
assert len(img.shape) == 4, "Image should be 4-dimensional"
assert img.shape[0] == 6 and img.shape[1] == 6, "First two dimensions should be 6x6"

# Create a figure with 6x6 subplots
fig, axes = plt.subplots(6, 6, figsize=(15, 15))

# Plot each image in the grid
for i in range(6):  # vertical
    for j in range(6):  # horizontal
        axes[i, j].imshow(img[i, j], cmap='gray')
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig(dir_results / "grid_visualization.png")
plt.close()

# Perform max projection over columns (axis=1)
max_projected = np.max(img, axis=1)  # This gives us 6 images
print(f"Max projected shape: {max_projected.shape}")

# Create a figure with 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot each max-projected image in the 2x3 grid
for idx, ax in enumerate(axes.flat):
    ax.imshow(max_projected[idx], cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.savefig(dir_results / "img_final.png")
plt.close()
