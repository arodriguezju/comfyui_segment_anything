

import os
import sys

#50 epcok, 4 repeats, 10 images. Only 20 epocs neede == 800 steps.
#73 imaegs, 1 repeat, only 11 epocs needed == 803 steps
def create_training_script(train_folder, model_path):
    command = f"""accelerate launch 
    --num_cpu_threads_per_process=2 
    "./kohya_ss/train_network.py"
    --enable_bucket 
    --min_bucket_reso=256 
    --max_bucket_reso=2048
    --pretrained_model_name_or_path="{model_path}" 
    --train_data_dir="{train_folder}" 
    --resolution="1024,1024" 
    --output_dir="/workspace/output_training" 
    --network_alpha=128
    --save_model_as=safetensors 
    --network_module=networks.lora 
    --text_encoder_lr=0.00005
    --unet_lr=0.0001
    --network_dim=1024
    --output_name="epicr-sand-earring-augmBg1024-1024-lr0.0001-0.00005-a128" 
    --max_train_epochs=11
    --no_half_vae 
    --lr_scheduler="constant" 
    --train_batch_size="1" 
    --save_every_n_epochs="3" 
    --mixed_precision="bf16" 
    --save_precision="bf16" 
    --cache_latents 
    --cache_latents_to_disk 
    --optimizer_type="Adafactor" 
    --optimizer_args relative_step=False scale_parameter=False warmup_init=False 
    --max_data_loader_n_workers="0" 
    --bucket_reso_steps=64 
    --gradient_checkpointing 
    --bucket_no_upscale 
    --noise_offset=0.0
    --xformers
    """
    #  --network_train_unet_only
    print(command)
    print(" ".join(line.strip() for line in command.splitlines()))

    return " ".join(line.strip() for line in command.splitlines())

def install_kohya():
    os.system(f"git clone  --branch v22.6.2 rhttps://github.com/bmaltais/kohya_ss.git")
    os.chdir("kohya_ss")
    os.system(f"pip install -r requirements.txt")
    os.system(f"pip install bitsandbytes")
    os.system(f"apt-get update")
    os.system(f"apt-get install python3-tk")
    os.chdir("..")

def install_grounddino():
    os.system(f"git clone https://github.com/arodriguezju/comfyui_segment_anything.git")
    os.chdir("comfyui_segment_anything")
    os.system(f"pip install -r requirements.txt")
    os.chdir("..")

def download_image(image_url):
    os.system(f"wget {image_url}")

def download_model(model_name, train_dir):    
    if not os.path.isfile(os.path.join(train_dir, model_name)):
        model_url = model_paths_maps[model_name]
        os.system(f"wget -P {train_dir} {model_url} --content-disposition")
    

def generate_training_data(image_name, train_dir):
     os.system(f"python comfyui_segment_anything/detect.py {image_name} 1024 output")

def train(image_folder, model_path):
    absolute_image_folder = os.path.abspath(image_folder)
    absolute_model_folder = os.path.abspath(model_path)
    training_command = create_training_script(absolute_image_folder, absolute_model_folder)
    os.system(training_command)

model_paths_maps = {
    "epicrealism_pureEvolutionV5.safetensors": "https://civitai.com/api/download/models/134065",
}

if __name__ == "__main__":
    train_folder = "output"
    model = "epicrealism_pureEvolutionV5.safetensors"
    model_path = os.path.abspath(os.path.join(train_folder, model))
    train_path = os.path.abspath(os.path.join(train_folder, "train_img"))

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    image_url = "https://huggingface.co/datasets/crom87/test-upload/resolve/main/IMG_1377.jpeg"#sys.argv[1]
    # image_url = "https://huggingface.co/datasets/crom87/test-upload/resolve/main/img2hd.png"#sys.argv[1]

    install_kohya()
    install_grounddino()
    download_image(image_url)
    download_model(model, train_folder)
    image_name = image_url.split("/")[-1]
    generate_training_data(image_name, train_folder)
    train(train_path, model_path)
    #TODO: Upload to huggung face
   