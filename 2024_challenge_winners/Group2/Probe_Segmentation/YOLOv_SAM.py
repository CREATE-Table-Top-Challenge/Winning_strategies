import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import argparse

def load_model(model_file):
    device = "cuda"
    model = YOLO(model_file)  # Load YOLO model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    return model, predictor, device

def process_image(image_path, model, predictor, device):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return None, None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # Perform detection
    results = model(image_path)
    # mask = np.zeros_like(image)  # Initialize mask as black image
    mask = np.zeros((1, image.shape[0], image.shape[1]), dtype=np.float32)
    for result in results:
        boxes = result.boxes
        bbox = boxes.xyxy.tolist()
        if (len(bbox)==0):
            return image, mask  # Return black image if no detections
        bbox = bbox[0]

        input_boxes = torch.tensor(bbox, device=device)  # bbox should be already a list or properly formatted before this conversion
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes.unsqueeze(0), image.shape[:2])
        mask, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        mask = mask.squeeze(0).cpu().numpy()

    return image, mask  # Return the original image and the mask

def save_masked_images(image, mask, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def show_mask(mask,ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # Ensure the mask tensor is on CPU before multiplying
    ax.imshow(mask_image)

def process_directory(directory, model, predictor, device, output_dir):
    all_masks = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_masked.png')
            image, masks = process_image(image_path, model, predictor, device)
            # save_masked_images(image, masks, output_path)
            all_masks.append(np.squeeze(masks))  # Squish masks and append to all_masks
    
    all_masks = np.array(all_masks)
    folder_name = os.path.split(directory)[-1]  # Extract the last folder name
    npy_filename = os.path.join(output_dir, folder_name + '_segmentations.npy')
    np.save(npy_filename, all_masks)  # Save all_masks as npy file

    return all_masks
    
    

def process_csv(csv_file, model, predictor, device, output_dir):
    data = pd.read_csv(csv_file)
    all_masks = []
    for _, row in data.iterrows():
        image_path = os.path.join(row['Folder'], row['FileName'])
        output_path = os.path.join(output_dir, os.path.splitext(row['FileName'])[0] + '_masked.png')
        image, masks = process_image(image_path, model, predictor, device)
        if masks:
            save_masked_images(image, masks, output_path)
            all_masks.append(masks)  # Collect all masks
    return np.array(all_masks)

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model, predictor, device = load_model(args.saved_model)

    if args.mode == 'csv':
        all_masks = process_csv(args.path, model, predictor, device, args.output_dir)
    elif args.mode == 'dir':
        all_masks = process_directory(args.path, model, predictor, device, args.output_dir)

    print(f"Processed masks shape: {all_masks.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images for segmentation.')
    parser.add_argument('--saved_model', type=str, default = 'yolov8-ultrasound.pt', help="Specify the saved model path")
    parser.add_argument('--mode', type=str, default='dir', choices=['csv', 'dir'], help='Choose "csv" for CSV file input or "dir" for directory input')
    parser.add_argument('--path', type=str, default=r'..\\dataset\\Test_Data\\Test_05', help='Path to the CSV file or directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to save the output masked images')
    args = parser.parse_args()
    main(args)
