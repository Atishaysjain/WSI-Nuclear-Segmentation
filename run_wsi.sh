python run_infer.py \
--gpu='0,1' \
--nr_types=5 \
--type_info_path=/content/drive/MyDrive/MoNuSAC/HoVer-Net/GitHub_Repo/hover_net_github/type_info.json \
--batch_size=64 \
--model_mode=fast \
--model_path=/content/drive/MyDrive/MoNuSAC/HoVer-Net/hovernet_fast_monusac_type_tf2pytorch.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
wsi \
--input_file=/content/drive/MyDrive/MoNuSAC/HoVer-Net/Data/MoNuSAC_images_and_annotations/MoNuSAC_images_and_annotations/TCGA-5P-A9K0-01Z-00-DX1/TCGA-5P-A9K0-01Z-00-DX1_1.svs/ \
--output_dir=/content/drive/MyDrive/MoNuSAC/HoVer-Net/Output/dummy_testing_folder/test_on_tif/ \
--save_thumb \
--save_mask

