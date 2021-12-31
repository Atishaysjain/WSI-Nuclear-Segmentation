python extract_arguments.py \
--gpu='0,1' \
--nr_types=5 \
--type_info_path=/content/drive/MyDrive/MoNuSAC/HoVer-Net/GitHub_Repo/hover_net_github/type_info.json \
--batch_size=7 \
--model_mode=fast \
--model_path=/content/drive/MyDrive/MoNuSAC/HoVer-Net/hovernet_fast_monusac_type_tf2pytorch.tar \
--nr_inference_workers=7 \
--nr_post_proc_workers=7 \
tile \
--input_file=/content/drive/MyDrive/MoNuSAC/HoVer-Net/Output/dummy_testing_folder/png_256_patches_TCGA-55-1594-01Z-00-DX1 \
--output_dir=/content/drive/MyDrive/MoNuSAC/HoVer-Net/Output/dummy_testing_folder/tile_test_output_on_png_256_TCGA-55-1594-01Z-00-DX1 \
--mem_usage=0.1 \
--draw_dot \
--save_qupath