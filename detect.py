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

backgrounds_folder = os.path.join(script_dir, "backgrounds")
masks_folder = os.path.join(script_dir, "masks")

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
    print("masks")
    # print(masks[0].size)
    return ToPILImage()(images[0].squeeze(0).permute(2, 0, 1))

def image_to_color_mask(image, green_threshold, red_threshold, blue_threshold):
    # Convert the PIL image to an OpenCV image (numpy array)
    open_cv_image = np.array(image)
    
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Isolate each channel
    blue_channel = open_cv_image[:, :, 0]
    green_channel = open_cv_image[:, :, 1]
    red_channel = open_cv_image[:, :, 2]

    # Calculate dominance
    green_dominance = green_channel - (0.5 * red_channel + 0.5 * blue_channel)
    red_dominance = red_channel - (0.5 * green_channel + 0.5 * blue_channel)
    blue_dominance = blue_channel - (0.5 * green_channel + 0.5 * red_channel)

    print(green_dominance, red_dominance, blue_dominance)
    # Create binary masks for each color dominance
    _, green_mask = cv2.threshold(green_dominance, green_threshold, 255, cv2.THRESH_BINARY)
    _, red_mask = cv2.threshold(red_dominance, red_threshold, 255, cv2.THRESH_BINARY)
    _, blue_mask = cv2.threshold(blue_dominance, blue_threshold, 255, cv2.THRESH_BINARY)

    # Combine masks to get a mask where any of the colors is dominant below its threshold
    combined_mask = np.maximum(np.maximum(green_mask, red_mask), blue_mask)

    # Convert the combined mask back to a PIL image
    mask_image = Image.fromarray(combined_mask.astype(np.uint8))

    return mask_image

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


def image_to_bw_mask(image):
    grayscale_image = image.convert("L")

    _, binary_image = cv2.threshold(np.array(grayscale_image), 0, 255, cv2.THRESH_BINARY)

    return Image.fromarray(binary_image)


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


def fill_regions_with_background_using_mask(image, mask_image, backgrounds_folder_path, repeats, train_class):
    # Load the main image and mask image, convert them to arrays
    main_image_array = np.array(image.convert('RGBA'))
    mask_array = np.array(mask_image.convert('L'))  # Convert mask to grayscale

    # Loop through each image in the backgrounds folder
    for index, background_image_name in enumerate(os.listdir(backgrounds_folder_path)):
        background_image_path = os.path.join(backgrounds_folder_path, background_image_name)
        
        try:
            # Open background image, resize it to match the main image, and convert to array
            print("Resizing to ", image.size)
            background_image = Image.open(background_image_path).resize(image.size).convert('RGBA')
            background_image_array = np.array(background_image)

            # Calculate the blend ratio based on the mask value
            blend_ratio = mask_array / 255.0
            blend_ratio = blend_ratio[:, :, None]  # Make it 3D for broadcasting

            # Blend the images based on the mask
            blended_image_array = (main_image_array * blend_ratio) + (background_image_array * (1 - blend_ratio))

            # Save the blended image to the training folder
            save_image_in_train_folder(Image.fromarray(blended_image_array.astype('uint8')).convert('RGB'), repeats, train_class)

        except Exception as e:
            print(f"Error processing {background_image_name}: {e}")

    # Or return filled_image to work with it directly in Python
    # return filled_image

def erode_mask(mask, erode_px, iterations):
    # Convert PIL image to OpenCV format
    image = np.array(mask)
    
    # Create erosion kernel
    kernel = np.ones((erode_px*2+1, erode_px*2+1), np.uint8)
    
    # Erode image
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    
    # Convert back to PIL format
    eroded_image_pil = Image.fromarray(cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2RGB))
    
    return eroded_image_pil

def apply_mask_to_image(mask_pil, rgb_pil):
    # Ensure the mask is in mode "L" for grayscale
    mask = mask_pil.convert("L")
    
    # Check if aspect ratios are the same
    # mask_aspect_ratio = mask.size[0] / mask.size[1]
    # rgb_aspect_ratio = rgb_pil.size[0] / rgb_pil.size[1]
    
    # if mask_aspect_ratio != rgb_aspect_ratio:
    #     raise ValueError(f"The mask and the RGB image have different aspect ratios. {mask_aspect_ratio} {rgb_aspect_ratio}")
    
    # Resize mask if necessary
    if mask.size != rgb_pil.size:
        mask = mask.resize(rgb_pil.size, Image.LANCZOS)
    
    new_image = Image.new("RGB", rgb_pil.size, (0, 0, 0, 0))
    
    # Apply the mask
    masked_image = Image.composite(rgb_pil, new_image, mask)
    
    return masked_image


if __name__ == "__main__":
    image_path = sys.argv[1]
    crop_resolution = int(sys.argv[2])

    # image_path = "ComfyUI_temp_upbiy_00029_.png"  # Path to the input image
    grounding_dino_model_name = "GroundingDINO_SwinT_OGC (694MB)"  # GroundingDINO model name
    sam_model_name = "sam_hq_vit_h (2.57GB)"  # SAM model name
    segmentation_class = "earring"  # Class to segment
    threshold = 0.5  # Detection threshold
    output_folder = sys.argv[3]  # Path for the output image
    train_repeats = 1
    train_class = "TOKstyle earring"


    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    torch_box = detect_box(grounding_dino_model_name, image, segmentation_class, threshold)

    cropped_image = crop_and_resize(image, torch_box, crop_resolution)
    cropped_image.save(os.path.join(output_folder, f"crop_{crop_resolution}.jpg"))

    save_image_in_train_folder(cropped_image, train_repeats, train_class)

    sam_masked_image = segment(sam_model_name, image, torch_box)
    sam_masked_image.save(os.path.join(output_folder, "masked_original.jpg"))

    sam_cropped_masked_image = crop_and_resize(sam_masked_image, torch_box, crop_resolution)

    #Masks
    sam_cropped_mask_bw_image = image_to_bw_mask(sam_cropped_masked_image)
    sam_cropped_mask_bw_image.save(os.path.join(output_folder, "crop_mask_sam.jpg"))
    fill_regions_with_background_using_mask(cropped_image, sam_cropped_mask_bw_image, backgrounds_folder, train_repeats, train_class)
    
    cropped_mask_bw_image_eroded = erode_mask(sam_cropped_mask_bw_image, 1, 3)
    fill_regions_with_background_using_mask(cropped_image, cropped_mask_bw_image_eroded, backgrounds_folder, train_repeats, train_class)
    
    #!!Use original size for crops/segmentation. Be sure we don't lose quality because of crop. Only resize later. 
    for filename in os.listdir(masks_folder):
        file_path = os.path.join(masks_folder, filename)
        try:  
            mask = Image.open(file_path).convert("RGB")
            fill_regions_with_background_using_mask(cropped_image, mask, backgrounds_folder, train_repeats, train_class)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    exit()
    cropped_mask_bw_image_eroded = erode_mask(cropped_mask_bw_image, 1, 3)
    cropped_mask_bw_image_eroded.save(os.path.join(output_folder, "crop_mask_bw_eroded.jpg"))

    cropped_mask_bw_image_eroded2 = erode_mask(cropped_mask_bw_image, 1, 5)
    cropped_mask_bw_image_eroded2.save(os.path.join(output_folder, "crop_mask_bw_eroded2.jpg"))

    external_mask1 = Image.open("masks/ComfyUI_temp_gobsl_00041_.png")
    external_mask2 = Image.open("masks/ComfyUI_temp_jgrpp_00001_.png")

    #Masked
    crop_masked = sam_cropped_masked_image
    crop_masked.save(os.path.join(output_folder, "crop_masked.jpg"))

    crop_masked_eroded = apply_mask_to_image(cropped_mask_bw_image_eroded, cropped_image)
    crop_masked_eroded.save(os.path.join(output_folder, "crop_masked_eroded.jpg"))

    crop_masked_eroded2 = apply_mask_to_image(cropped_mask_bw_image_eroded2, cropped_image)
    crop_masked_eroded2.save(os.path.join(output_folder, "crop_masked_eroded2.jpg"))

    crop_masked_external1 = apply_mask_to_image(external_mask1, cropped_image)
    crop_masked_external1.save(os.path.join(output_folder, "crop_masked_external1.jpg"))

    crop_masked_external2 = apply_mask_to_image(external_mask2, cropped_image)
    crop_masked_external2.save(os.path.join(output_folder, "crop_masked_external2.jpg"))

    
    cropped_masked_green_image = image_to_color_mask(cropped_masked_image, 20, 255, 255)

    canny = canny_edge_detector(cropped_image, 1)


    cropped_masked_green_image.save(os.path.join(output_folder, "crop_masked_green.jpg"))

    image.save(os.path.join(output_folder, "original.jpg"))
    canny.save(os.path.join(output_folder, f"canny_{crop_resolution}.jpg"))

    
    save_image_in_train_folder(cropped_image, train_repeats, train_class)

    # augmented_images = 3

    train_repeats_for_augmented = train_repeats

    # blue_aug = augment_with_background_color(cropped_masked_image, 0, 0, 255)
    # save_image_in_train_folder(blue_aug, train_repeats_for_augmented, train_class + " over a blue background")
    # green_aug = augment_with_background_color(cropped_masked_image, 0, 255, 0)
    # save_image_in_train_folder(green_aug, train_repeats_for_augmented, train_class + " over a green background")
    # red_aug = augment_with_background_color(cropped_masked_image, 255, 0, 0)
    # save_image_in_train_folder(red_aug, train_repeats_for_augmented, train_class + " over a red background")
    
    # fill_black_regions_with_backgrounds_efficient(cropped_masked_image, "backgrounds", train_repeats_for_augmented, train_class)



    # detect_and_segment_boxes(image_path, grounding_dino_model_name, sam_model_name, segmentation_class, threshold, crop_resolution, train_repeats, train_class, output_folder)
