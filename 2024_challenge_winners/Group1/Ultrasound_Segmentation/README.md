# SAM - based few shot ultrasound detection 

We do few shot ultrasound detection using SAM: 
   1. Prompt SAM with the ultrasound bounding box to generate mask predictions (`generate_sam_masks_from_boxes.py`)
   2. Finetune SAM with no prompt, to reconstruct its own prompted predictions (`train_sam_ultrasound_segmentation.py`)
   3. Generate new predictions on test data (`generate_test_predictions_sam.ipynb`)

## Setup

Make your python environment - follow original challenge page to set up the basic environment, then use:
```
pip install segment-anything-py
```
to install SAM, and go to https://github.com/facebookresearch/segment-anything and follow instructions to download a SAM checkpoint.

## Label generation

(Replace paths based on your file system configuration)
```
python generate_sam_masks_from_boxes.py 
   --data_csv_file /home/paulw/Challenge_Code/uncompressed_data/Training_Data/Training_Data.csv
   --target_dir /home/paulw/Challenge_Code/sam_ultrasound_masks
   --tool_name ultrasound 
```


## Training 
Run training using the following command (use the cli to change configuration as desired, try running `--help`)

```
python train_sam_ultrasound_segmentation.py \
    --sam_weights /home/paulw/Challenge_Code/sam_vit_b_01ec64.pth \
    --sam_type vit_b \
    --splits_file splits_trainee.json \
    --freeze_image_encoder \
    --batch_size 2 \
    --lr 1e-5 \
    --use_amp \
    --epochs 10 \
    --save_dir checkpoints/ultrasound_detect/$(date "+%Y-%m-%d_%H:%M") \
    --data_csv_path /home/paulw/Challenge_Code/uncompressed_data/Training_Data/Training_Data.csv \
    --masks_dir /home/paulw/Challenge_Code/sam_ultrasound_masks 
```

## Testing 
To create the test predictions, follow through the `generate_test_predictions_sam.ipynb` notebook. Alternatively, run the test script like follows: 

```bash
python test_sam_ultrasound_segmentation.py \
    --train_checkpoint /home/paulw/Challenge_Code/checkpoints/ultrasound_detect/2024-04-20_14:37/vit_b_best.pt \
    --data_csv_path /home/paulw/Challenge_Code/uncompressed_data/Training_Data/Training_Data.csv \
    --splits_file splits_trainee.json \
    --target_dir /home/paulw/Challenge_Code/checkpoints/ultrasound_detect/2024-04-20_14:37/test 
```

This test script will create both `.png` outputs and `.mp4` videos and put them in `target_dir`.