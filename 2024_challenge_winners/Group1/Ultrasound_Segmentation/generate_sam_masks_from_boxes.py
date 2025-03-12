import argparse
import os
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
import pandas as pd
import json
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        'Use the existing SAM model to generate `ground truth` masks based on bounding boxes'
    )
    parser.add_argument(
        "--data_csv_file",
        help="Path to csv file containing box annotations and file locations",
    )
    parser.add_argument("--target_dir", help="Path where the masks will be saved")
    parser.add_argument(
        "--tool_name",
        help="which tool to generate masks from bounding boxes for",
        default="ultrasound",
    )
    parser.add_argument(
        "--max_num",
        help="If set, caps the number of total predictions (use for debugging)",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite existing masks in the target directory.",
    )
    return parser.parse_args()


@torch.no_grad()
def main(args):
    # setup target dir
    os.makedirs(args.target_dir, exist_ok=True)

    # setup sam model
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    data_table = pd.read_csv(args.data_csv_file)
    for i, row in tqdm(data_table.iterrows(), total=len(data_table)):
        if args.max_num is not None and i > args.max_num:
            break

        filename = row["FileName"]
        folder = row["Folder"]
        fullpath = os.path.join(folder, filename)

        output_filename = os.path.join(
            args.target_dir,
            args.tool_name.lower() + "_" + filename.replace(".jpg", ".png"),
        )
        if not args.overwrite and os.path.exists(output_filename):
            continue

        image = cv2.imread(fullpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        annotations = row["Tool bounding box"]
        annotations = json.loads(annotations.replace("'", '"'))

        target_tool_box = None
        for box_info in annotations:
            label = box_info["class"]
            if label.lower() == args.tool_name.lower():
                target_tool_box = box_info

        if target_tool_box is not None:
            xmin = target_tool_box["xmin"]
            xmax = target_tool_box["xmax"]
            ymin = target_tool_box["ymin"]
            ymax = target_tool_box["ymax"]

            input_box = np.array([xmin, ymin, xmax, ymax])

            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
        else:
            masks = np.zeros((1, image.shape[0], image.shape[1]), bool)
            input_box = None

        masks = masks[0].astype("uint8") * 255
        cv2.imwrite(output_filename, masks)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
