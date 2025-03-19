import numpy as np
import os
join = os.path.join
import pandas as pd
from skimage import io, transform, segmentation
import torch
import random
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient#, hausdorff_distance, extract_points
import argparse
import cv2
import matplotlib.pyplot as plt







# Set up the parser
parser = argparse.ArgumentParser(description='Preprocess grey and RGB images for segmentation')
parser.add_argument('--image_size', type=int, default=256, help='image size for resizing')
parser.add_argument('--seed', type=int, default=2023, help='Random seed for reproducibility')
parser.add_argument('--model_type', type=str, default='vit_b', help='Type of model for embedding extraction')  
parser.add_argument('--original_checkpoint', type=str, default='SAM/sam_vit_b_01ec64.pth', help='SAM weights')  
parser.add_argument('--tuned_checkpoint', type=str, default='sam_model_best.pth', help='finetuned model')  
parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation')  
parser.add_argument('--data_ratio', type=float, default=1, help='Fraction of the data to use (between 0 and 1)')

args = parser.parse_args()


# Paths
base_path = os.getcwd()
segmentation_path = os.path.join(base_path, "Submit")
os.makedirs(segmentation_path, exist_ok=True)

# Set random seed
random.seed(args.seed)

# Initialize SAM model
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)  

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
 


# %% Collect image paths from specified folders only
data_path = os.path.join('Test', 'Test_Data')
for part in [data_path]:
    part_path = os.path.join(base_path, part)
    image_paths = []
    for video_folder in os.listdir(part_path):
        video_path = os.path.join(part_path, video_folder)
        if os.path.isdir(video_path):
            images = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')]
            image_paths.extend(images)

        # Optionally select a portion of the data
        if args.data_ratio < 1.0:
            selected_size = int(len(image_paths) * args.data_ratio)
            image_paths = random.sample(image_paths, selected_size)
        
        # Process each selected image
        imgs = []
                
        for img_path in tqdm(image_paths, total=len(image_paths)):
            image_name = os.path.basename(img_path)
            
            # Load and preprocess image
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
            
           
            
            imgs.append(image_data_pre)
            
        np.save(join(segmentation_path, f'{video_folder}_images'), np.array(imgs))
        print(f"Video {video_folder} processing completed.")
        
        

#%


        model_type = args.model_type
        checkpoint = args.checkpoint
        orig_checkpoint = args.orig_checkpoint
        device = args.device
        tuned_sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        
        
        model_save_path = base_path
        model_path = join(model_save_path, checkpoint)
        
        state_dict = torch.load(model_path)
        
        tuned_sam_model.load_state_dict(state_dict)
        
        
               
        sam_trans = ResizeLongestSide(tuned_sam_model.image_encoder.img_size)
        
        
               
        def get_bbox_from_mask(mask):
            '''Returns a bounding box from a mask'''
            y_indices, x_indices = np.where(mask > 0)
            y_min = y_max = x_min = x_max = 0  # Default values
            if y_indices.shape[0] > 0:
                y_min, y_max = np.min(y_indices), np.max(y_indices)
            if x_indices.shape[0] > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
            
            
            # add perturbation to bounding box coordinates
            H, W = mask.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        
            return np.array([x_min, y_min, x_max, y_max])
        
        ori_sam_segs = []
        medsam_segs = []
        bboxes = []
        for img in tqdm(imgs, total=len(imgs)):
                        
            # predict the segmentation mask using the fine-tuned model
            H, W = img.shape[:2]
            resize_img = sam_trans.apply_image(img)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = tuned_sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
            with torch.no_grad():
                image_embedding = tuned_sam_model.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
                                
                sparse_embeddings, dense_embeddings = tuned_sam_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
                medsam_seg_prob, _ = tuned_sam_model.mask_decoder(
                    image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
                    image_pe=tuned_sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                    multimask_output=False,
                    )
                medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
                # convert soft mask to hard mask
                medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
                medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
                medsam_segs.append(medsam_seg)
                
                
                    
               
                # ## visualize the segmentation masks
                # _, axs = plt.subplots(1, 2, figsize=(25, 25))
                # axs[0].imshow(cv2.resize(img, (480, 640), interpolation=cv2.INTER_CUBIC))
                

                # axs[0].axis('off')
                
                    
                # axs[1].imshow(cv2.resize(img, (480, 640), interpolation=cv2.INTER_CUBIC))
                # show_mask(cv2.resize(medsam_seg, (480, 640), interpolation=cv2.INTER_CUBIC), axs[1])

                # axs[1].axis('off')
                # plt.show()  
                # plt.subplots_adjust(wspace=0.01, hspace=0)
                
                # plt.close()
                
                
        
        np.save(join(segmentation_path, f'{video_folder}_segmentations'), np.array(medsam_segs))
        print(f"Video {video_folder} segmentations are saved.")
            
            
            


            
