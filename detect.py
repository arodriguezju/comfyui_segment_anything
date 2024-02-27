import os
import sys
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import cv2
from torchvision.transforms import ToPILImage
import random

# Ensure the script's directory is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the necessary functions from the provided code snippet
from node import load_groundingdino_model, load_sam_model, groundingdino_predict, sam_segment

def detect_box(grounding_dino_model_name, image, segmentation_class, threshold):

    def extract_biggest_box(boxes):
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        biggest_box_idx = torch.argmax(box_areas)
        return boxes[biggest_box_idx]
    
    grounding_dino_model = load_groundingdino_model(grounding_dino_model_name)
    item = image.convert('RGBA')
    boxes = groundingdino_predict(grounding_dino_model, item, segmentation_class, threshold)
    print(f"Detected {boxes.size(dim=0)} boxes.")
    return extract_biggest_box(boxes)

def segment(sam_model_name, image, torch_box):
    sam_model = load_sam_model(sam_model_name)
    item = image.convert('RGBA')
    (images, masks) = sam_segment(
                sam_model,
                item,
                torch_box
            )
    #TODO: Check if more than one mask or if any
    return ToPILImage()(images[0].squeeze(0).permute(2, 0, 1))

def crop_and_resize(image, box, crop_resolution):

    image = image.crop(box.numpy())

    #resize keeping original highest dimension as crop_resolution
    width, height = image.size
    if width > height:
        new_width = crop_resolution
        new_height = int(height * (crop_resolution / width))
    else:
        new_height = crop_resolution
        new_width = int(width * (crop_resolution / height))
    return image.resize((new_width, new_height))

    # image.save(os.path.join(output_folder, f"crop_{crop_resolution}.jpg"))

    # canny = canny_edge_detector(image, 1)
    # canny.save(os.path.join(output_folder, f"canny_{crop_resolution}.jpg"))



def detect_and_segment_boxes(image_path, grounding_dino_model_name, sam_model_name, segmentation_class, threshold, crop_resolution,  train_repeats, train_class, output_folder):
    
    def fillWithColor(image):
        image_np = np.array(image)

        # Generate a random RGB color
        random_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)

        # Find all black pixels (where all RGB values are 0)
        black_pixels_mask = (image_np[:, :, 0] == 0) & (image_np[:, :, 1] == 0) & (image_np[:, :, 2] == 0)

        # Replace black pixels with the random color
        image_np[black_pixels_mask] = random_color

        # Convert the modified array back to a PIL image
        return Image.fromarray(image_np)

    
    # Load models
    grounding_dino_model = load_groundingdino_model(grounding_dino_model_name)
    sam_model = load_sam_model(sam_model_name)

    # Load and process the image
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)

     #create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image.save(os.path.join(output_folder, "original.jpg"))
    item = image.convert('RGBA')
    # # Predict boxes using GroundingDINO model
    boxes = groundingdino_predict(grounding_dino_model, item, segmentation_class, threshold)
    print(boxes)
    # Create an ImageDraw object to draw on the image

    print(f"Detected {boxes.size(dim=0)} boxes.")

    if boxes.size(0) == 0:
        return
    
    
    torch_biggest_box = extract_biggest_box(boxes)

    (images, masks) = sam_segment(
                sam_model,
                item,
                torch_biggest_box
            )

    # Convert the mask to a PIL image
    for i, image in enumerate(images):
        print(f"Saving mask {image.shape}.")
        pil_image = ToPILImage()(image.squeeze(0).permute(2, 0, 1))
        pil_image = fillWithColor(pil_image)
        pil_image.save(os.path.join(output_folder, f"mask_{i}.jpg"))
        # save_image(image.squeeze(0), os.path.join(output_folder, "masked.jpg"))

    biggest_box = torch_biggest_box.numpy()

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

def augment_with_background_color(image, red, blue, green):
    image_np = np.array(image)
    black_pixels_mask = (image_np[:, :, 0] == 0) & (image_np[:, :, 1] == 0) & (image_np[:, :, 2] == 0)
    image_np[black_pixels_mask] = [red, blue, green]
    return Image.fromarray(image_np)
    
def save_image_in_train_folder(image, repeats, name):
    train_img_folder = os.path.join(output_folder, "train_img")
    if not os.path.exists(train_img_folder):
        os.makedirs(train_img_folder)

    train_class_joined = "_".join(name.split(" "))
    image_class_folder = os.path.join(train_img_folder, f"{repeats}_{train_class_joined}")
    if not os.path.exists(image_class_folder):
        os.makedirs(image_class_folder)
    
    image.save(os.path.join(image_class_folder, f"{random.randint(0, 9999)}.jpg"))


def fill_black_regions_with_backgrounds_efficient(image, backgrounds_folder_path, repeats, train_class):
    # Load the image with black regions
    main_image_array = np.array(image.convert('RGBA'))

    # Identify black pixels (or nearly black, if needed)
    black_pixels_mask = np.all(main_image_array[:, :, :3] == 0, axis=-1)

    # Loop through each image in the backgrounds folder
    for index, background_image_name in enumerate(os.listdir(backgrounds_folder_path)):
        background_image_path = os.path.join(backgrounds_folder_path, background_image_name)
        
        try:
            # Open background image, resize it to match the main image, and convert to array
            background_image = Image.open(background_image_path).resize(image.size).convert('RGBA')
            background_image_array = np.array(background_image)

            # Use the mask to replace black pixels in the main image with those from the background
            main_image_array[black_pixels_mask] = background_image_array[black_pixels_mask]
            save_image_in_train_folder(Image.fromarray(main_image_array).convert('RGB'), repeats, train_class)

        except Exception as e:
            print(f"Error processing {background_image_name}: {e}")

    
    # Or return filled_image to work with it directly in Python
    # return filled_image



if __name__ == "__main__":
    image_path = sys.argv[1]
    crop_resolution = int(sys.argv[2])

    # image_path = "ComfyUI_temp_upbiy_00029_.png"  # Path to the input image
    grounding_dino_model_name = "GroundingDINO_SwinT_OGC (694MB)"  # GroundingDINO model name
    sam_model_name = "sam_hq_vit_b (379MB)"  # SAM model name
    segmentation_class = "earring"  # Class to segment
    threshold = 0.5  # Detection threshold
    output_folder = sys.argv[3]  # Path for the output image
    train_repeats = 5
    train_class = "TOKstyle earring"


    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    torch_box = detect_box(grounding_dino_model_name, image, segmentation_class, threshold)
    masked_image = segment(sam_model_name, image, torch_box)
    cropped_image = crop_and_resize(image, torch_box, crop_resolution)
    cropped_masked_image = crop_and_resize(masked_image, torch_box, crop_resolution)

    canny = canny_edge_detector(cropped_image, 1)



    masked_image.save(os.path.join(output_folder, "masked.jpg"))
    image.save(os.path.join(output_folder, "original.jpg"))
    cropped_image.save(os.path.join(output_folder, f"crop_{crop_resolution}.jpg"))
    canny.save(os.path.join(output_folder, f"canny_{crop_resolution}.jpg"))

    
    # save_image_in_train_folder(cropped_image, train_repeats, train_class)

    # augmented_images = 3

    train_repeats_for_augmented = train_repeats

    # blue_aug = augment_with_background_color(cropped_masked_image, 0, 0, 255)
    # save_image_in_train_folder(blue_aug, train_repeats_for_augmented, train_class + " over a blue background")
    # green_aug = augment_with_background_color(cropped_masked_image, 0, 255, 0)
    # save_image_in_train_folder(green_aug, train_repeats_for_augmented, train_class + " over a green background")
    # red_aug = augment_with_background_color(cropped_masked_image, 255, 0, 0)
    # save_image_in_train_folder(red_aug, train_repeats_for_augmented, train_class + " over a red background")
    
    fill_black_regions_with_backgrounds_efficient(cropped_masked_image, "backgrounds", train_repeats_for_augmented, train_class)



    # detect_and_segment_boxes(image_path, grounding_dino_model_name, sam_model_name, segmentation_class, threshold, crop_resolution, train_repeats, train_class, output_folder)
