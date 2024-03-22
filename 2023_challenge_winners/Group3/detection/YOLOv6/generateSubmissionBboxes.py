import os
import json
import errno
import argparse
import numpy as np
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=str, required=True)
    parser.add_argument("--label-csv", type=str, default="results/test_labels.csv")
    parser.add_argument("--classname-json", type=str, default="data/class_names.json")
    parser.add_argument("--img-size", nargs="+", type=int, default=[480, 640])
    
    args = parser.parse_args()
    return args


def convert_labels_to_csv(
    label_dir="runs/inference/exp/test/labels",
    label_csv="results/test_labels.csv",
    classname_json="data/class_names.json",
    img_size=640
):
    # Make save directory, if it doesn't exist
    os.makedirs(os.path.dirname(label_csv), exist_ok=True)

    # Get list of classes corresponding to each integer output
    if os.path.exists(classname_json) and os.path.getsize(classname_json) > 0:
        with open(classname_json, "r") as f:
            classes = json.load(f)
            print(classes)
            print(f"Loaded class names from {classname_json}")
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), classname_json)

    # If int given for img_size, convert to list
    if isinstance(img_size, int):
        img_size = [img_size, img_size]

    # Iterate through each .txt file and convert YOLOv6 output to submission format
    fn_list = []
    bbox_list = []
    label_fns = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".txt")]
    for label_fn in label_fns:
        # Get filename of image
        frame_fn = os.path.splitext(os.path.basename(label_fn))[0] + ".jpg"
        fn_list.append(frame_fn)

        # Get bounding box information
        bbox = []
        with open(label_fn, "r") as f:
            for line in f:
                bbox_data = line.rstrip().split()  # remove trailing characters and convert to list of strings

                # Convert relative coordinates to pixels
                pixel_center_x = float(bbox_data[1]) * img_size[1]
                pixel_center_y = float(bbox_data[2]) * img_size[0]
                pixel_width = float(bbox_data[3]) * img_size[1]
                pixel_height = float(bbox_data[4]) * img_size[0]

                # Get bbox corner coordinates in pixels
                x_min = round(pixel_center_x - pixel_width / 2)
                y_min = round(pixel_center_y - pixel_height / 2)
                x_max = round(pixel_center_x + pixel_width / 2)
                y_max = round(pixel_center_y + pixel_height / 2)

                # Add bbox data to a dictionary and append to list
                bbox_dict = {
                    "class": classes[int(bbox_data[0])],
                    "xmin": x_min,
                    "ymin": y_min,
                    "xmax": x_max,
                    "ymax": y_max,
                    "conf": float(bbox_data[5])
                }
                bbox.append(bbox_dict)
        bbox_list.append(bbox)

    # Create and save dataframe from lists of filenames and bounding boxes
    df = pd.DataFrame({
        "FileName": fn_list,
        "Tool bounding box": bbox_list
    })
    df.to_csv(label_csv, index=False)
    print(f"Saved bounding box csv file to {label_csv}.")


def main(args):
    convert_labels_to_csv(**vars(args))


if __name__ == "__main__":
    args = get_parser()
    main(args)
