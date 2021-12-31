python run_infer.py \
--gpu='0,1' \
--nr_types=0 \
--type_info_path='' \
--batch_size=64 \
--model_mode=original \
--model_path=/content/drive/MyDrive/Colab_Notebooks/pretrained/hover-net-pytorch-weights/hovernet_original_kumar_notype_tf2pytorch.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
wsi \
--input_dir=/content/drive/MyDrive/Colab_Notebooks/Dataset/MoNuSeg/MoNuSegTestData/ \
--output_dir=/content/drive/MyDrive/Colab_Notebooks/Output/MoNuSeg/MoNuSegTestData_output/ \
--save_thumb \
--save_mask
