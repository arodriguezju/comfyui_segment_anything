import os
import sys
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import cv2

# Ensure the script's directory is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the necessary functions from the provided code snippet
from node import load_groundingdino_model, load_sam_model, groundingdino_predict, sam_segment

def detect_and_segment_boxes(image_path, grounding_dino_model_name, sam_model_name, segmentation_class, threshold, crop_resolution,  train_repeats, train_class, output_folder):
   
    def extract_biggest_box(boxes):
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        biggest_box_idx = torch.argmax(box_areas)
        return boxes[biggest_box_idx]
    # Load models
    grounding_dino_model = load_groundingdino_model(grounding_dino_model_name)
    # sam_model = load_sam_model(sam_model_name)

    # Load and process the image
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)

     #create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image.save(os.path.join(output_folder, "original.jpg"))

    # # Predict boxes using GroundingDINO model
    boxes = groundingdino_predict(grounding_dino_model, image.convert("RGBA"), segmentation_class, threshold)
    print(boxes)
    # Create an ImageDraw object to draw on the image

    print(f"Detected {boxes.size(dim=0)} boxes.")

    if boxes.size(0) == 0:
        return
    
    biggest_box = extract_biggest_box(boxes).numpy()
    print("Using biggest box: ", biggest_box)

    # crop image to bigges box
    image = image.crop(biggest_box)

    #resize keeping original smallest dimension as crop_resolution
    width, height = image.size
    if width < height:
        new_width = crop_resolution
        new_height = int(height * (crop_resolution / width))
    else:
        new_height = crop_resolution
        new_width = int(width * (crop_resolution / height))
    image = image.resize((new_width, new_height))

    image.save(os.path.join(output_folder, "crop.jpg"))

    canny = canny_edge_detector(image, 1)
    canny.save(os.path.join(output_folder, "canny.jpg"))


    train_img_folder = os.path.join(output_folder, "train_img")
    if not os.path.exists(train_img_folder):
        os.makedirs(train_img_folder)

    train_class_joined = "_".join(train_class.split(" "))
    image_class_folder = os.path.join(train_img_folder, f"{train_repeats}_{train_class_joined}")
    if not os.path.exists(image_class_folder):
        os.makedirs(image_class_folder)
    image.save(os.path.join(image_class_folder, f"{train_class_joined}.jpg"))





def canny_edge_detector(image, dilation):
    # Read the image

    # PIL image to opencv image 
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Check if image is loaded
    if image is None:
        print("Error: Image not found.")
        return

    # Apply Canny edge detection
    edges = cv2.Canny(image, 250, 300)


    # Dilate the edges to make them thicker
    kernel = np.ones((dilation, dilation), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    dilated_edges_rgb = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2RGB)


    # return PIL image
    return Image.fromarray(dilated_edges_rgb)

if __name__ == "__main__":
    image_path = sys.argv[1]
    crop_resolution = int(sys.argv[2])

    # image_path = "ComfyUI_temp_upbiy_00029_.png"  # Path to the input image
    grounding_dino_model_name = "GroundingDINO_SwinT_OGC (694MB)"  # GroundingDINO model name
    sam_model_name = "sam_hq_vit_b (379MB)"  # SAM model name
    segmentation_class = "earring"  # Class to segment
    threshold = 0.5  # Detection threshold
    output_folder = sys.argv[3]  # Path for the output image
    train_repeats = 20
    train_class = "TOKstyle silver oxidized earring"

    detect_and_segment_boxes(image_path, grounding_dino_model_name, sam_model_name, segmentation_class, threshold, crop_resolution, train_repeats, train_class, output_folder)
