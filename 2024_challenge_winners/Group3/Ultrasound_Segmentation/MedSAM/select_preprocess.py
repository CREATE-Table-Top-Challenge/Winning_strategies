import numpy as np
import os
import pandas as pd
from skimage import io, transform
import torch
import random
from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import cv2
import glob

def extract_ultrasound_boxes(csv_file, image_name):
    df = pd.read_csv(csv_file)
    row = df[df['FileName'] == image_name]
    if not row.empty:
        tool_bboxes = eval(row['Tool bounding box'].iloc[0])
        return [bbox for bbox in tool_bboxes if bbox['class'] == 'ultrasound']
    return []

parser = argparse.ArgumentParser(description='Preprocess grey and RGB images for segmentation')
parser.add_argument('--sample_size', type=int, default=100, help='Number of frames to select for retraining')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data')
parser.add_argument('--image_size', type=int, default=256, help='image size for resizing')
parser.add_argument('--seed', type=int, default=2023, help='Random seed for reproducibility')
parser.add_argument('--model_type', type=str, default='vit_b', help='Type of model for embedding extraction')
parser.add_argument('--checkpoint', type=str, default='SAM/sam_vit_b_01ec64.pth', help='Model checkpoint')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation')

args = parser.parse_args()

base_path = os.getcwd()
segmentation_path = os.path.join(base_path, "segmentation_mask")

random.seed(args.seed)

sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

video_folders = {}
for part in [1, 2, 3, 6, 7, 8, 9, 10]: #train parts
# for part in [4, 5]: #test parts
    part_path = os.path.join(base_path, f"Training_Data_Part{part}")
    for video_folder in os.listdir(part_path):
        video_path = os.path.join(part_path, video_folder)
        if os.path.isdir(video_path):
            images = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')]
            csv_file = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.csv')][0]
            video_folders[video_folder] = (images, csv_file)

total_videos = len(video_folders)
samples_per_video = args.sample_size // total_videos
extra_samples = args.sample_size % total_videos

# Maintain a list to ensure x% empty bboxes
selected_images = []
selected_csvs = []
empty_bbox_images = []
empty_bbox_csvs = []

for video_folder, (images, csv_file) in video_folders.items():
    if len(images) < samples_per_video:
        sampled_images = images
    else:
        sampled_images = random.sample(images, samples_per_video + (1 if extra_samples > 0 else 0))
        extra_samples -= 1
    
    for img in sampled_images:
        bboxes = extract_ultrasound_boxes(csv_file, os.path.basename(img))
        if bboxes == []:
            empty_bbox_images.append(img)
            empty_bbox_csvs.append(csv_file)
        else:
            selected_images.append(img)
            selected_csvs.append(csv_file)

# Adjust selection to meet the x% empty bbox requirement
required_empty_bbox_count = int(0.3 * args.sample_size)
current_empty_count = len(empty_bbox_images)

if current_empty_count < required_empty_bbox_count:
    deficit = required_empty_bbox_count - current_empty_count
    additional_empty = random.sample([i for i, img in enumerate(selected_images) if extract_ultrasound_boxes(selected_csvs[i], os.path.basename(img)) == []], deficit)
    for idx in additional_empty:
        empty_bbox_images.append(selected_images.pop(idx))
        empty_bbox_csvs.append(selected_csvs.pop(idx))
elif current_empty_count > required_empty_bbox_count:
    excess = current_empty_count - required_empty_bbox_count
    to_remove = random.sample(range(current_empty_count), excess)
    for idx in sorted(to_remove, reverse=True):
        empty_bbox_images.pop(idx)
        empty_bbox_csvs.pop(idx)

selected_images += empty_bbox_images
selected_csvs += empty_bbox_csvs





# Process each selected image
imgs = []
gts = []
img_embeddings = []

for img_path, csv_file in tqdm(zip(selected_images, selected_csvs), total=len(selected_images)):
    image_name = os.path.basename(img_path)
    
    bboxes = extract_ultrasound_boxes(csv_file, image_name)
    
    seg_mask_path = os.path.join(segmentation_path, image_name.replace('.jpg', '.npy'))
    if os.path.exists(seg_mask_path):
        try:
            gt_data = np.load(seg_mask_path)
        except:
            continue
        
        image_data = io.imread(img_path)
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # Image preprocessing
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
        image_data_pre[image_data == 0] = 0
        image_data_pre = cv2.resize(image_data_pre, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)
        image_data_pre = np.uint8(image_data_pre)
        
        
        gt_data_resized = transform.resize(gt_data, (args.image_size, args.image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=False)
        gt_data_resized = np.uint8(gt_data_resized > 0.5)  # Binarizing the resized ground truth

       
        imgs.append(image_data_pre)
        if bboxes==[]:
            gts.append(np.uint8(np.zeros(gt_data_resized.shape)))
        else:
            gts.append(gt_data_resized)
        # Embedding calculation
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image_data_pre)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(args.device)
        input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
        with torch.no_grad():
            embedding = sam_model.image_encoder(input_image)
            img_embeddings.append(embedding.cpu().numpy()[0])
    else:
        print(f"Warning: No segmentation mask found for {image_name}")


# Save to npz
np.savez_compressed('train_data.npz', imgs=imgs, gts=gts, embeddings=img_embeddings)  # Data saving


print("Dataset processing completed. Train and test datasets are saved.")

