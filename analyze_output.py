import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import openslide
import json
import pandas as pd

def merge_json(proc_mag, output_dir, json_path_wsi):
    list_json_files = os.listdir(os.path.join(output_dir, "json"))
    data = []

    nuc_count = 1
    merged_dict = {"mag" : proc_mag, "nuc" : {}}
    nuc_list = []

    patch_coords_df = pd.read_csv(os.path.join(output_dir, "wsi_info", "patch_coordinates.csv"))

    for f in list_json_files:

        patch_name = f.split(".")[0]
        # print(patch_name)
        tl_x = int(patch_coords_df["tl_x"].loc[patch_coords_df["patch_name"] == patch_name+".png"].to_numpy()[0])
        tl_y = int(patch_coords_df["tl_y"].loc[patch_coords_df["patch_name"] == patch_name+".png"].to_numpy()[0])
        # print(type(tl_x))

        with open(os.path.join(output_dir, "json", f), "rb") as infile:

            infile = json.load(infile)

            for key, value in infile["nuc"].items():

                # print(value)
                
                value["bbox"][0][0] = value["bbox"][0][0] + tl_x
                value["bbox"][0][1] = value["bbox"][0][1] + tl_y
                value["bbox"][1][0] = value["bbox"][1][0] + tl_x
                value["bbox"][1][1] = value["bbox"][1][1] + tl_y

                value["centroid"][0] = value["centroid"][0] + tl_x
                value["centroid"][1] = value["centroid"][1] + tl_y

                for contour_coor_index in range(0, len(value["contour"])):
                    value["contour"][contour_coor_index][0] = value["contour"][contour_coor_index][0] + tl_x
                    value["contour"][contour_coor_index][1] = value["contour"][contour_coor_index][1] + tl_y

                nuc_list.append((str(nuc_count), value))
                nuc_count += 1

                # print(value)
                # break
    
    merged_dict["nuc"] = dict(nuc_list)

    # print(merged_dict)

    with open(json_path_wsi, "w") as outfile:    
        json.dump(merged_dict, outfile)

def get_output_file(proc_mag, output_dir, n = None):

    json_output_files = []
    [json_output_files.append(os.path.join(output_dir, 'json', output_file)) for output_file in os.listdir(os.path.join(output_dir, 'json'))]

    if(n != None):
        json_path_wsi = json_output_files[n]
    else:
        wsi_name = output_dir.split("/")[-1]
        wsi_json_filename = f"{wsi_name}_json.json"
        json_path_wsi = os.path.join(output_dir, "wsi_info", wsi_json_filename)
        # if not os.path.exists(json_path_wsi):
        #     merge_json(proc_mag, output_dir, json_path_wsi)
        merge_json(proc_mag, output_dir, json_path_wsi)

    return json_path_wsi


def get_spatial_info(df, bbox_list, nucei_type):

    epithelial_width = []
    Lymphocyte_width = []
    Macrophage_width = []
    Neutrophil_width = []
    nolabel_width = []

    epithelial_height = []
    Lymphocyte_height = []
    Macrophage_height = []
    Neutrophil_height = []
    nolabel_height = []

    for i in range(0, len(df)):

        curr_nuclei_bbox_list = bbox_list[i]
        min_x = 1e+5
        max_x = 0
        min_y = 1e+5
        max_y = 0
        for j in range(0, len(curr_nuclei_bbox_list)):
            x = curr_nuclei_bbox_list[j][0]
            y = curr_nuclei_bbox_list[j][1]
            min_x = min(x, min_x)
            max_x = max(x, max_x)
            min_y = min(y, min_y)
            max_y = max(y, max_y)
        width = max_x - min_x
        height = max_y - min_y
        
        # if(df['type'] == '0'):
        #     nolabel_width.append(width)
        #     nolabel_height.append(height)
        # print(df.iloc[i]['type'])
        if(df.iloc[i]['type'] == 1):
            epithelial_width.append(width)
            epithelial_height.append(height)
        elif(df.iloc[i]['type'] == 2):
            Lymphocyte_width.append(width) 
            Lymphocyte_height.append(height)
        elif(df.iloc[i]['type'] == 3):
            Macrophage_width.append(width)
            Macrophage_height.append(height)
        elif(df.iloc[i]['type'] == 4):
            Neutrophil_width.append(width)
            Neutrophil_height.append(height)


    if(nucei_type == 0):
        return sum(nolabel_width)/len(nolabel_width), sum(nolabel_height)/len(nolabel_height), nolabel_width, nolabel_height

    if(nucei_type == 1):
        return (sum(epithelial_width)/len(epithelial_width)), (sum(epithelial_height)/len(epithelial_height)), epithelial_width, epithelial_height

    if(nucei_type == 2):
        return sum(Lymphocyte_width)/len(Lymphocyte_width), sum(Lymphocyte_height)/len(Lymphocyte_height), Lymphocyte_width, Lymphocyte_height

    if(nucei_type == 3):
        return sum(Macrophage_width)/len(Macrophage_width), sum(Macrophage_height)/len(Macrophage_height), Macrophage_width, Macrophage_height

    if(nucei_type == 4):
        return sum(Neutrophil_width)/len(Neutrophil_width), sum(Neutrophil_height)/len(Neutrophil_height), Neutrophil_width, Neutrophil_height

def info_output(proc_mag, output_dir, n = None):

    json_path_wsi = get_output_file(proc_mag, output_dir, n = None)

    bbox_list_wsi = []
    centroid_list_wsi = []
    contour_list_wsi = [] 
    type_list_wsi = []

    with open(json_path_wsi, 'r') as json_file:
        print(json_path_wsi)
        data = json.load(json_file)
        mag_info = data['mag']
        nuc_info = data['nuc']
        for inst in nuc_info:
            inst_info = nuc_info[inst]
            inst_centroid = [round(inst_info['centroid'][0], 2), round(inst_info['centroid'][1], 2)]
            centroid_list_wsi.append(inst_centroid)
            # inst_contour = inst_info['contour']
            # contour_list_wsi.append(inst_contour)
            inst_bbox = inst_info['bbox']
            bbox_list_wsi.append(inst_bbox)
            inst_type = inst_info['type']
            type_list_wsi.append(inst_type)
        
    # slide_info_df = pd.DataFrame({'centroid':centroid_list_wsi, 'contour':contour_list_wsi, 'type':type_list_wsi})
    slide_info_df = pd.DataFrame({'centroid':centroid_list_wsi, 'type':type_list_wsi})
    # print(slide_info_df)
    slide_info = slide_info_df.groupby('type')

    # if(len(slide_info_df.where(slide_info_df['type'] == '0')) > 0):
    #   nolabel_df = slide_info.get_group('0')
    #   n_nolabel = len(nolabel_df)
    #   nolabel_average_width, nolabel_average_height, nolabel_df['width'], nolabel_df['height'] = get_spatial_info(slide_info_df, bbox_list_wsi, 0)

    # pd.DataFrame
    epithelial_df = pd.DataFrame()
    Lymphocyte_df = pd.DataFrame()
    Macrophage_df = pd.DataFrame()
    Neutrophil_df = pd.DataFrame()

    epithelial_average_width=-1
    epithelial_average_height=-1
    Lymphocyte_average_width=-1
    Lymphocyte_average_height=-1
    Macrophage_average_width=-1
    Macrophage_average_height=-1
    Neutrophil_average_width=-1
    Neutrophil_average_height=-1

    if(len(slide_info_df.where(slide_info_df['type'] == 1).dropna()) > 0):
        # print(slide_info_df.where(slide_info_df['type'] == 1).dropna())
        # print(len(slide_info_df[slide_info_df.where(slide_info_df['type'] == 1)]))
        epithelial_df = slide_info.get_group(1)
        n_epithelial = len(epithelial_df)
        epithelial_average_width, epithelial_average_height, epithelial_df['width'], epithelial_df['height'] = get_spatial_info(slide_info_df, bbox_list_wsi, 1)

    if(len(slide_info_df.where(slide_info_df['type'] == 2).dropna()) > 0):
        Lymphocyte_df = slide_info.get_group(2)
        n_Lymphocyte = len(Lymphocyte_df)
        Lymphocyte_average_width, Lymphocyte_average_height, Lymphocyte_df['width'], Lymphocyte_df['height'] = get_spatial_info(slide_info_df, bbox_list_wsi, 2)

    if(len(slide_info_df.where(slide_info_df['type'] == 3).dropna()) > 0):
        Macrophage_df = slide_info.get_group(3)
        n_Macrophage = len(Macrophage_df)
        Macrophage_average_width, Macrophage_average_height, Macrophage_df['width'], Macrophage_df['height'] = get_spatial_info(slide_info_df, bbox_list_wsi, 3)

    if(len(slide_info_df.where(slide_info_df['type'] == 4).dropna()) > 0):
        Neutrophil_df = slide_info.get_group(4)
        n_Neutrophil = len(Neutrophil_df)
        Neutrophil_average_width, Neutrophil_average_height, Neutrophil_df['width'], Neutrophil_df['height'] = get_spatial_info(slide_info_df, bbox_list_wsi, 4)

    return epithelial_average_width, epithelial_average_height, epithelial_df, Lymphocyte_average_width, Lymphocyte_average_height, Lymphocyte_df, Macrophage_average_width, Macrophage_average_height, Macrophage_df, Neutrophil_average_width, Neutrophil_average_height, Neutrophil_df

