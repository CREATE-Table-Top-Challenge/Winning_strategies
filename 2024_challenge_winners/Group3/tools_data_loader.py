# Import dependencies
import pandas as pd
import glob
import cv2
from torch.utils.data import Dataset

class ToolsDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.samples = []
        
        labels_csv_path = glob.glob(f'{dir}/' + '*_Labels.csv', recursive=True)
        
        if not labels_csv_path:
            print("No label csv file found in the directory!")
            return
        
        labels = pd.read_csv(labels_csv_path[0], sep=',')
        
        if labels.shape[0] == 0:
            print("No records found in the label csv file!")
            return
        
        for _, frame_info in labels.iterrows():
            img_path = f'{dir}/{frame_info["FileName"]}'
            box_dict = eval(frame_info['Tool bounding box'])
            
            for box_info in box_dict:
                self.samples.append((img_path, box_info))
 
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box_info = self.samples[idx]
        cropped_box = self.crop_box(img_path, box_info)
        label = box_info['class']
        
        return cropped_box, label
    
    def crop_box(self, img_path, box_dict):
        img = cv2.imread(img_path)
        
        xmin = box_dict['xmin']
        ymin = box_dict['ymin']
        xmax = box_dict['xmax']
        ymax = box_dict['ymax']
        
        cropped_box = img[ymin:ymax, xmin:xmax]
        
        return cropped_box