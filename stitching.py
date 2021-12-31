import h5py
import numpy as np
import os
import pdb
from PIL import Image
import math
import cv2
import time

def DrawGrid(img, coord, shape, thickness=2, color=(0,0,0,255)):
    img = cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img

def DrawMapFromCoords(patch_path, canvas, wsi, coord, patch_size, vis_level, downsamples, indices=None, verbose=1, draw_grid=False):

    patch = cv2.imread(patch_path)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    coord = np.ceil(coord / downsamples).astype(np.int32)
    canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
    canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
    if draw_grid:
        canvas = DrawGrid(canvas, coord, patch_size)

    return canvas

def get_raw_heatmap(wsi, level, alpha = 0.4, bg_color=(0,0,0)):

    w, h = wsi.level_dimensions[level]

    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    heatmap = np.array(heatmap)

    return heatmap 

def save_heatmap(heatmap, hovernet_output, wsi_name):

    heatmap = Image.fromarray(heatmap)
    stitch_path = os.path.join(hovernet_output, wsi_name+"_overlayed.png")
    heatmap.save(stitch_path)
    stitch_path = os.path.join(hovernet_output, wsi_name+"_overlayed.tif")
    heatmap.save(stitch_path)
    stitch_path = os.path.join(hovernet_output, wsi_name+"_overlayed.jpg")
    rgb_heatmap = heatmap.convert('RGB')
    rgb_heatmap.save(stitch_path)
