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
--input_file=/content/drive/MyDrive/Colab_Notebooks/Dataset/MoNuSAC_Testing_Data_TIF_Images/test_folder_14th_oct/ \
--output_dir=/content/hover_net/output/hovernet_MoNuSac_TCGA_2Z_A9JG_01Z_00_DX1/ \
--save_thumb \
--save_mask

