

import os
import sys

def create_training_script(train_folder, model_path):
    return """accelerate launch 
    --num_cpu_threads_per_process=2 
    "./train_network.py"
    --enable_bucket 
    --min_bucket_reso=256 
    --max_bucket_reso=2048
    --pretrained_model_name_or_path="${model_path}" 
    --train_data_dir="${train_folder}" 
    --resolution="512,512" 
    --output_dir="/workspace/output_training" 
    --network_alpha="1" 
    --save_model_as=safetensors 
    --network_module=networks.lora 
    --text_encoder_lr=0.0004 
    --unet_lr=0.0004 
    --network_dim=256 
    --output_name="last" 
    --lr_scheduler_num_cycles="10" 
    --no_half_vae 
    --learning_rate="0.0004" 
    --lr_scheduler="constant" 
    --train_batch_size="1" 
    --save_every_n_epochs="1" 
    --mixed_precision="fp16" 
    --save_precision="fp16" 
    --cache_latents 
    --cache_latents_to_disk 
    --optimizer_type="Adafactor" 
    --optimizer_args relative_step=False scale_parameter=False warmup_init=False 
    --max_data_loader_n_workers="0" 
    --bucket_reso_steps=64 
    --gradient_checkpointing 
    --xformers 
    --bucket_no_upscale 
    --noise_offset=0.0
"""

if __name__ == "__main__":
    absolute_image_folder = sys.argv[1]  
    absolute_model_path = sys.argv[2]     
    training_command = create_training_script(absolute_image_folder, absolute_model_path)
    os.system(training_command)