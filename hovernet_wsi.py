import os 
import argparse
import sys
import h5py
import openslide
import numpy as np
import cv2
import subprocess
import json
import torch
import logging
import copy
import pandas as pd
from misc.utils import log_info
from docopt import docopt
import extract_arguments
import analyze_output
import stitching

def create_run_wsi(input_file, output_dir):
  with open ('run_wsi_standard.sh', 'r') as f:
    f_contents = f.readlines()
    for index, field in enumerate(f_contents):
      if(len(field) >= 12 and field[:12] == '--input_file'):
        input_index = index
      if(len(field) >= 12 and field[:12] == '--output_dir'):
        output_index = index
    with open ('run_wsi.sh', 'w') as nf:
      f.seek(0)
      nf.seek(0)
      for line_no, line in enumerate(f):
        if(line_no == input_index):
          new_input = "--input_file=" + str(input_file) + "/ \\\n"
          nf.write(new_input)
        elif(line_no == output_index):
          new_output = "--output_dir=" + str(output_dir) + "/ \\\n"
          nf.write(new_output)
        else:
          nf.write(line)

def create_run_tile(input_file, output_dir):
  with open ('run_tile_standard.sh', 'r') as f:
    f_contents = f.readlines()
    for index, field in enumerate(f_contents):
      if(len(field) >= 12 and field[:12] == '--input_file'):
        input_index = index
      if(len(field) >= 12 and field[:12] == '--output_dir'):
        output_index = index
    with open ('run_tile.sh', 'w') as nf:
      f.seek(0)
      nf.seek(0)
      for line_no, line in enumerate(f):
        if(line_no == input_index):
          new_input = "--input_file=" + str(input_file) + "/ \\\n"
          nf.write(new_input)
        elif(line_no == output_index):
          new_output = "--output_dir=" + str(output_dir) + "/ \\\n"
          nf.write(new_output)
        else:
          nf.write(line)

parser = argparse.ArgumentParser()

parser.add_argument("--input_wsi_dir", type=str, default="/content/drive/MyDrive/MoNuSAC/HoVer-Net/Data/WSI/", help="path to directory containing input wsi")

parser.add_argument("--segmentation_output_dir", type=str, default="/content/drive/MyDrive/MoNuSAC/HoVer-Net/Output/Segmentation_Output/", help="path to the directory having segmentation output created by extract_features_fp.py")

parser.add_argument("--hovernet_wsi_output", type=str, default="/content/drive/MyDrive/MoNuSAC/HoVer-Net/Output/wsi_output/", help="path to the directory which will store the output of the hovernet code")

parser.add_argument("--patch_size", type=int, default=1024, help="size of the patch upon which hovernet code would operate")

parser.add_argument("--level", type=int, default=0, help="level of wsi at which patches will be extracted for the hovernet code")

parser.add_argument("--draw_grid", type=bool, default=False, help="If you want a grid on the overlay then enter True else False (default : False)")

parser.add_argument("--save_patch_overlay", type=bool, default=False, help="If you want to save the overlay png image of each individual patch then enter True else False (default : False)")

parser.add_argument("--save_mat", type=bool, default=True, help="If you want to save a .mat file for each individual patch containing nuclei information then enter True else False (default : True). These files can be helpful for generation of training data.")

args = parser.parse_args()

if __name__ == '__main__':

  input_wsi_dir = args.input_wsi_dir
  segmentation_output_dir = args.segmentation_output_dir
  hovernet_wsi_output = args.hovernet_wsi_output
  patch_size = args.patch_size
  level = args.level
  draw_grid = args.draw_grid
  save_patch_overlay = args.save_patch_overlay
  save_mat = args.save_mat
  
  wsi_list = os.listdir(input_wsi_dir) # a list of wsi images

  for wsi_file in wsi_list:

    print(f"For {wsi_file}")
    
    input_wsi_file = os.path.join(input_wsi_dir, wsi_file)
    wsi_name = wsi_file[:-4] 

    segmentation_patch = os.path.join(segmentation_output_dir, "patches", wsi_name + ".h5") # path of segmentation path hdf5 file for the given wsi
    
    if not os.path.exists(os.path.join(hovernet_wsi_output, wsi_name)):
      os.mkdir(os.path.join(hovernet_wsi_output, wsi_name))
    hovernet_output = os.path.join(hovernet_wsi_output, wsi_name) # path of the hovernet output folder for the given wsi

    with h5py.File(segmentation_patch, "r") as f:
      coords = f['coords']
      coords = np.array(coords)
    
    count = 0 ## Change

    create_run_tile(input_file = input_wsi_file, output_dir = hovernet_output) # This will generate a run_tile.sh file that will contain the information for running the hovernet code

    subprocess.run('chmod +x /content/drive/MyDrive/MoNuSAC/HoVer-Net/GitHub_Repo/hover_net_pipeline/run_tile.sh', shell = True) # Modifying the permission of run_tile.sh so that we can execute it
    hovernet_arguments = subprocess.run('/content/drive/MyDrive/MoNuSAC/HoVer-Net/GitHub_Repo/hover_net_pipeline/run_tile.sh', shell = True, capture_output =True)
    hovernet_arguments_str = hovernet_arguments.stdout.decode()
    sub_cmd = extract_arguments.get_sub_cmd(hovernet_arguments_str)
    arguments, sub_args = extract_arguments.get_dict(hovernet_arguments_str, sub_cmd)
    method_args, run_args = extract_arguments.get_arguments(arguments, sub_args, sub_cmd)
    
    if sub_cmd == 'tile':
        from infer.tile import InferManager
        infer = InferManager(**method_args)
    else:
        from infer.wsi import InferManager
        infer = InferManager(**method_args)

    if not os.path.exists("/content/drive/MyDrive/MoNuSAC/HoVer-Net/GitHub_Repo/hover_net_pipeline/temp_patch/" + wsi_name):
      os.mkdir("/content/drive/MyDrive/MoNuSAC/HoVer-Net/GitHub_Repo/hover_net_pipeline/temp_patch/" + wsi_name)

    wsi = openslide.open_slide(input_wsi_file)

    # sys.path.insert(1, '/path/to/application/app/folder')
    # sys.path.append("/content/drive/MyDrive/MoNuSAC/HoVer-Net/GitHub_Repo/hover_net_pipeline/patches_utils/wsi_core")
    # from WholeSlideImage import WholeSlideImage
    # WSI_object = WholeSlideImage(input_wsi_file)

    if sub_cmd == 'tile':

        if not os.path.exists(hovernet_output + '/json/'):
          os.mkdir(hovernet_output + '/json/')
        if not os.path.exists(hovernet_output + '/mat/'):
          os.mkdir(hovernet_output + '/mat/')
        if not os.path.exists(hovernet_output + '/overlay/'):
          os.mkdir(hovernet_output + '/overlay/')
        if run_args["save_qupath"]:
          if not os.path.exists(hovernet_output + '/qupath/'):
            os.mkdir(hovernet_output + "/qupath/")
          

        patch_coords_df = pd.DataFrame(columns=["patch_name", "tl_x", "tl_y", "w", "h"])
        patch_name_list = []
        tl_x_list = []
        tl_y_list = []
        patch_width_list = []
        patch_height_list = []

        heatmap = stitching.get_raw_heatmap(wsi, level, alpha = 0.4, bg_color=(0,0,0))
        downsamples = wsi.level_downsamples[level]
        patch_size_tuple = tuple((np.array((patch_size, patch_size)) * downsamples).astype(np.int32))
        patch_size_draw_coords = tuple(np.ceil((np.array(patch_size_tuple)/np.array(downsamples))).astype(np.int32))

        for index, coord in enumerate(coords):
            if(index != 0):
                if os.path.exists(patch_path):
                    os.remove(patch_path)
            tl_x = coord[0]
            tl_y = coord[1]
            patch = wsi.read_region((tl_x, tl_y), level, (patch_size, patch_size))
            patch_name = f"patch_{index}.png"
            patch_path = "/content/drive/MyDrive/MoNuSAC/HoVer-Net/GitHub_Repo/hover_net_pipeline/temp_patch/" + wsi_name + "/" + patch_name
            run_args["input_file"] = patch_path
            patch.save(patch_path) 
            print(f"patch {index} saved") 
            infer.process_file_list(run_args, save_mat)

            # Stitching the output
            overlay_path_path = os.path.join(hovernet_output, "overlay", patch_name)
            heatmap = stitching.DrawMapFromCoords(overlay_path_path, heatmap, wsi, coord, patch_size_draw_coords, level, downsamples, indices=None, draw_grid = False)

            if not save_patch_overlay:
                os.remove(overlay_path_path)

            #tlwh format
            patch_name_list.append(patch_name)
            tl_x_list.append(tl_x)
            tl_y_list.append(tl_y)
            patch_width_list.append(patch_size)
            patch_height_list.append(patch_size)

        stitching.save_heatmap(heatmap, hovernet_output, wsi_name)
        
        patch_coords_df["patch_name"] = patch_name_list
        patch_coords_df["tl_x"] = tl_x_list
        patch_coords_df["tl_y"] = tl_y_list
        patch_coords_df["w"] = patch_width_list
        patch_coords_df["h"] = patch_height_list
        if not os.path.exists(os.path.join(hovernet_output, "wsi_info")):
          os.mkdir(os.path.join(hovernet_output, "wsi_info"))
        if not os.path.exists(os.path.join(hovernet_output, "wsi_info", "patch_coordinates.csv")):
          patch_coords_df.to_csv(os.path.join(hovernet_output, "wsi_info", "patch_coordinates.csv"))

        # proc_mag = run_args["proc_mag"]
        proc_mag = None
        epithelial_average_width, epithelial_average_height, epithelial_df, Lymphocyte_average_width, Lymphocyte_average_height, Lymphocyte_df, Macrophage_average_width, Macrophage_average_height, Macrophage_df, Neutrophil_average_width, Neutrophil_average_height, Neutrophil_df = analyze_output.info_output(proc_mag, hovernet_output, None)

        epithelial_df.to_csv(os.path.join(hovernet_output, "wsi_info", "epithelial_df.csv"))
        Lymphocyte_df.to_csv(os.path.join(hovernet_output, "wsi_info", "Lymphocyte_df.csv"))
        Macrophage_df.to_csv(os.path.join(hovernet_output, "wsi_info", "Macrophage_df.csv"))
        Neutrophil_df.to_csv(os.path.join(hovernet_output, "wsi_info", "Neutrophil_df.csv"))


    else:
        print("Code for run_wsi has not been written yet")

  