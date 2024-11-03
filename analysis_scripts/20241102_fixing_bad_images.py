"""
python -m ipdb analysis_scripts/20241102_fixing_bad_images.py
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

# results dir
sys.path.insert(0, "..")
sys.path.insert(0, ".")
from benchmark.build_raw_dataset.download_data import download_csv

dir_results_parent = Path(__file__).parent / "results" / Path(__file__).stem
dir_results_parent.mkdir(exist_ok=True, parents=True)
from aicsimageio import AICSImage

##### form 12 #####
if 0:
    if 1:
        f = "../Downloads/form12/TS_PBTOhNGN2hiPSCs_BR3_N03_Unmixed-6Chan&6Time.tiff"
        suffix = "0"
    else:
        f = "../Downloads/form12/ZSTACK_PBTOhNGN2hiPSCs_BR1_N16_Unmixed-6Chan&6slice.tiff"
        suffix = "1"

    assert Path(f).exists()
    img0_ = AICSImage(f)
    data = img0_.data
    imgs = []
    num = 0

    # Determine the size of each sub-image (assuming all are the same size)
    img_height, img_width = data[
        0, 0, 0].shape  # Shape of each individual grayscale image
    nrows, ncols = data.shape[1:3]

    # Process each small image and append to the list
    for i in range(nrows):
        row_images = []
        for j in range(ncols):
            img = data[0, i, j]
            l, u = img.min(), img.max()
            img = (img - l) / (u - l)
            img = (img * 255).astype(np.uint8)
            row_images.append(Image.fromarray(img))
        imgs.append(row_images)

    # Create an empty image to hold the 6x7 grid
    grid_img = Image.new(
        'L', (img_width * ncols, img_height * nrows))  # 'L' mode for grayscale

    # Paste each image into the grid
    for i, row_images in enumerate(imgs):
        for j, img in enumerate(row_images):
            grid_img.paste(img, (j * img_width, i * img_height))

    # Save the final grid image
    grid_img.save(f"grid_image_{suffix}.png")

    ipdb.set_trace()

import numpy as np


def make_grid(images, nrow, ncol, padding=0, pad_value=1):
    # Verify we have enough images
    n_images = len(images)
    if n_images > nrow * ncol:
        raise ValueError(
            f"Too many images ({n_images}) for grid size {nrow}x{ncol}")

    # Get dimensions of the first image
    H, W = images[0].shape

    # Create grid filled with padding value
    grid_H = H * nrow + padding * (nrow - 1)
    grid_W = W * ncol + padding * (ncol - 1)
    grid = np.full((grid_H, grid_W), pad_value, dtype=images[0].dtype)

    # Place images in the grid
    idx = 0
    for i in range(nrow):
        for j in range(ncol):
            if idx >= len(images):
                break

            # Calculate position with padding
            top = i * (H + padding)
            left = j * (W + padding)

            # Place the image
            grid[top:top + H, left:left + W] = images[idx]
            idx += 1

    return grid


def norm_01(img):
    l, u = img.min(), img.max()
    img = (img - l) / (u - l)
    return img


##### form 12 #####
# first image
# img 0
if 0:
    f = "../Downloads/form12/TS_PBTOhNGN2hiPSCs_BR3_N03_Unmixed-6Chan&6Time.tiff"
    assert Path(f).exists()
    img0_ = AICSImage(f)
    data = img0_.data
    imgs = list(data[0, 3, :])
    imgs = [norm_01(im) for im in imgs]
    grid = make_grid(imgs, 3, 3, padding=10, pad_value=1)
    grid = (grid * 255).astype(np.uint8)
    # Image.fromarray(grid).save(f"grid_image_{suffix}.png")
    Image.fromarray(grid).save(
        f"TS_PBTOhNGN2hiPSCs_BR3_N03_Unmixed-6Chan&6Time.png")

if 0:
    f = "../Downloads/form12/ZSTACK_PBTOhNGN2hiPSCs_BR1_N16_Unmixed-6Chan&6slice.tiff"
    assert Path(f).exists()
    img0_ = AICSImage(f)
    data = img0_.data
    imgs = list(data[0, :, 3])
    imgs = [norm_01(im) for im in imgs]
    grid = make_grid(imgs, 2, 3, padding=10, pad_value=1)
    grid = (grid * 255).astype(np.uint8)
    # Image.fromarray(grid).save(f"grid_image_{suffix}.png")
    Image.fromarray(grid).save(
        f"ZSTACK_PBTOhNGN2hiPSCs_BR1_N16_Unmixed-6Chan&6slice.png")
    pass

##### form 76 #####
if 0:
    for f_in, f_out in zip(("APOE e2.tif", "APOE e3.tiff", "APOE e4.tiff"), ("APOE e2", "APOE e3", "APOE e4")):
        f = f"../Downloads/form76/{f_in}"
        assert Path(f).exists()
        img0_ = AICSImage(f)
        data = img0_.data
        imgs = list(data[0, :, 3])
        imgs = [norm_01(im) for im in imgs]
        grid = make_grid(imgs, 1, 2, padding=10, pad_value=1)
        grid = (grid * 255).astype(np.uint8)
        # Image.fromarray(grid).save(f"grid_image_{suffix}.png")
        Image.fromarray(grid).save(
            f"{f_out}.png")

##### form 78 #####
if 0:
    for f_in, f_out in zip(("APOE e3.tiff", "Rev APOE e2.tiff"), (("APOE e3", "Rev APOE e2"))):
        f = f"../Downloads/form78/{f_in}"
        assert Path(f).exists()
        img0_ = AICSImage(f)
        data = img0_.data
        imgs = list(data[0, :, 3])
        imgs = [norm_01(im) for im in imgs]
        grid = make_grid(imgs, 3, 3, padding=10, pad_value=1)
        grid = (grid * 255).astype(np.uint8)
        # Image.fromarray(grid).save(f"grid_image_{suffix}.png")
        Image.fromarray(grid).save(
            f"{f_out}.png")


##### form 78 #####
if 1:
    for f_in, f_out in zip(("APOE e3.tiff", "Rev APOE e4.tif"), (("APOE e3", "Rev APOE e4"))):
        f = f"../Downloads/form79/{f_in}"
        assert Path(f).exists()
        img0_ = AICSImage(f)
        data = img0_.data
        imgs = list(data[0, :, 3])
        imgs = [norm_01(im) for im in imgs]
        grid = make_grid(imgs, 3, 3, padding=10, pad_value=1)
        grid = (grid * 255).astype(np.uint8)
        # Image.fromarray(grid).save(f"grid_image_{suffix}.png")
        Image.fromarray(grid).save(
            f"{f_out}.png")
    ipdb.set_trace()
    pass




