import os
import requests
import torch
import numpy as np
import cv2
import supervision as sv
import pandas as pd
import base64
import time

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pathlib import Path



home_dir = os.getcwd()
weights_dir = os.path.join(home_dir, 'SAM')


os.makedirs(weights_dir, exist_ok=True)


file_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'


file_path = os.path.join(weights_dir, 'sam_vit_b_01ec64.pth')


mask_save_path = os.path.join(home_dir, 'segmentation_mask')

if not os.path.exists(mask_save_path):
    os.makedirs(mask_save_path)

# Download the file with requests
response = requests.get(file_url)
response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

# Write the file contents to the local file path
with open(file_path, 'wb') as file:
    file.write(response.content)

print(f'File downloaded to {file_path}')

#%%
CHECKPOINT_PATH = os.path.join(weights_dir, 'sam_vit_b_01ec64.pth')
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

#%%
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"

#%%
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

#%%

mask_predictor = SamPredictor(sam)





# Base path where the "Training_Data_PartX" directories are located
base_path = home_dir

# Function to extract bounding boxes for the ultrasound class
def extract_ultrasound_boxes(csv_file, image_name):
    
    df = pd.read_csv(csv_file)
    row = df[df['FileName'] == image_name]

    if not row.empty:
        tool_bboxes = eval(row['Tool bounding box'].iloc[0])
        return [bbox for bbox in tool_bboxes if bbox['class'] == 'ultrasound']
    return []



def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded

# Loop through each "Training_Data_PartX" directory
for i in range(1, 11):
    part_dir = f'Training_Data_Part{i}'
    part_path = os.path.join(base_path, part_dir)

    if os.path.exists(part_path):
        
        for subdir in os.listdir(part_path):
            subdir_path = os.path.join(part_path, subdir)
            
            
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            for csv_file in csv_files:
                csv_path = os.path.join(subdir_path, csv_file)
                images = [f for f in os.listdir(subdir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                
                for image in images:
                    image_path = os.path.join(subdir_path, image)
                    mask_file_path = os.path.join(mask_save_path, os.path.splitext(os.path.basename(image))[0] + '.npy')
                    
                   
                    if os.path.exists(mask_file_path):
                        print(f"Mask file already exists for {image}. Skipping mask generation.")
                        continue
                    
                    ultrasound_bboxes = extract_ultrasound_boxes(csv_path, image)
                    
                    if ultrasound_bboxes==[]:
                        ultrasound_bboxes = {'x': 0, 'y': 0, 'width': 0, 'height': 0, 'label': ''}
                    else: 
                        ultrasound_bboxes = {'x': ultrasound_bboxes[0]['xmin'], 'y': ultrasound_bboxes[0]['ymin'], 'width': ultrasound_bboxes[0]['xmax'], 'height': ultrasound_bboxes[0]['ymax'], 'label': ''}
                        
                    box = np.array([
                        ultrasound_bboxes['x'],
                        ultrasound_bboxes['y'],
                        ultrasound_bboxes['width'],
                        ultrasound_bboxes['height']
                    ])
                    
                    
                    
                    image_bgr = cv2.imread(image_path)
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    
                    mask_predictor.set_image(image_rgb)
                    
                    masks, scores, logits = mask_predictor.predict(
                        box=box,
                        multimask_output=False
                    )
                    
                    
                    
                    box_annotator = sv.BoxAnnotator(color=sv.Color.red())
                    mask_annotator = sv.MaskAnnotator(color=sv.Color.red(), color_lookup=sv.ColorLookup.INDEX)
                    
                    detections = sv.Detections(
                        xyxy=sv.mask_to_xyxy(masks=masks),
                        mask=masks
                    )
                    detections = detections[detections.area == np.max(detections.area)]
                    
                    source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
                    segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
                    
                    sv.plot_images_grid(
                        images=[source_image, segmented_image, masks[0]],
                        grid_size=(1, 3),
                        titles=['source image', 'segmented image', 'segmentation mask']
                    )
                    
                    
                    
                    np.save(mask_file_path, masks[0])
                    
                    
                    
                                        
                    

