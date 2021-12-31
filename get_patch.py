import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

def get_patch_from_coordinate(wsi_name, x, y, patch_size=1024):

    patch_coordinates_path = f"/content/drive/MyDrive/MoNuSAC/HoVer-Net/Output/wsi_output/{wsi_name}/wsi_info/patch_coordinates.csv"
    patch_coords_df = pd.read_csv(patch_coordinates_path)

    tl_x_array = np.array(patch_coords_df["tl_x"])
    tl_y_array = np.array(patch_coords_df["tl_y"])

    x_indices = []
    for i in range(0, len(tl_x_array)):
        if(tl_x_array[i]+patch_size >= x and tl_x_array[i] <= x):
            x_indices.append(i)
    
    y_indices = []
    for i in range(0, len(tl_y_array)):
        if(tl_y_array[i]+patch_size >= y and tl_y_array[i] <= y):
            y_indices.append(i)

    x_indices = set(x_indices)
    y_indices = set(y_indices)
    index = x_indices & y_indices
    index = list(index)[0]

    required_patch_name = patch_coords_df["patch_name"].iloc[index]
    patch_path = f"/content/drive/MyDrive/MoNuSAC/HoVer-Net/Output/wsi_output/{wsi_name}/overlay/{required_patch_name}"

    return patch_path

# parser = argparse.ArgumentParser()

# parser.add_argument("--wsi_name", type=str, default="C3L-01663-21", help="Name of the whole slide image")

# parser.add_argument("--x", type=int, help="x coordinate")

# parser.add_argument("--y", type=int, help="y coordinate")

# parser.add_argument("--patch_size", type=int, default=1024, help="size of the overlay patches")

# args = parser.parse_args()

if __name__ == '__main__':

    # wsi_name = args.wsi_name
    # x = args.x
    # y = args.y
    # patch_size = args.patch_size

    patch_path = get_patch_from_coordinate(wsi_name, x, y, patch_size)

    img = mpimg.imread(patch_path)
    plt.imshow(img)
    plt.show()
