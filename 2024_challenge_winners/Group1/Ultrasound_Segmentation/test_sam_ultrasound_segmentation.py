from train_sam_ultrasound_segmentation import (
    SamWrapper,
    ImagesAndUltrasoundMasksDataset,
)
from segment_anything.predictor import SamPredictor
from segment_anything.build_sam import sam_model_registry
import cv2
import matplotlib.pyplot as plt
import argparse
import torch
import os
from tqdm import tqdm
from glob import glob
import sys
import re
import shutil


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_argument_parser():
    parser = argparse.ArgumentParser("Test SAM ultrasound segmentation - generates predictions and videos.")
    parser.add_argument("--sam_weights", default="sam_vit_b_01ec64.pth")
    parser.add_argument(
        "--sam_type", choices=sam_model_registry.keys(), default="vit_b"
    )
    parser.add_argument(
        "--train_checkpoint", help="path to training checkpoint from train script"
    )
    parser.add_argument("--data_csv_path")
    parser.add_argument(
        "--splits_file", help="path to splits_file.json specifying test videos"
    )
    parser.add_argument("--split", help="Which split to evaluate on", default="val")
    parser.add_argument("--target_dir", help="where to write the test outputs")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to override results of previous tests",
    )
    parser.add_argument(
        "--n_videos_test",
        type=int,
        default=None,
        help="If you only want to test on a few videos, specify the number here. By default, tests on all videos",
    )

    return parser


def main(args):
    sam_model = sam_model_registry[args.sam_type](args.sam_weights)
    model = SamWrapper(sam_model)

    print(f"Loading checkpoint: {args.train_checkpoint}")
    model.load_state_dict(torch.load(args.train_checkpoint, map_location=DEVICE))
    model.eval().to(DEVICE)
    predictor = SamPredictor(model.model)

    _dataset = ImagesAndUltrasoundMasksDataset(
        args.data_csv_path, split=args.split, splits_file=args.splits_file
    )
    data_table = _dataset.data_table
    video_ids = list(data_table.VideoID.unique())
    if args.n_videos_test is not None:
        video_ids = video_ids[: args.n_videos_test]

    os.makedirs(args.target_dir, exist_ok=True)

    for video_idx, video_id in enumerate(video_ids):
        print(f"Processing video {video_idx}/{len(video_ids)}: {video_id}")

        subtable = data_table.loc[data_table.VideoID == video_id]
        video_filepaths = []
        for i, (_, row) in enumerate(tqdm(subtable.iterrows(), total=len(subtable))):
            filename = row["FileName"]
            folder = row["Folder"]
            fullpath = os.path.join(folder, filename)

            image = cv2.imread(fullpath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = get_mask_prediction(image, predictor)

            # write prediction
            output_filename = os.path.join(
                args.target_dir,
                args.split,
                video_id,
                "sam_prediction" + "_" + filename.replace(".jpg", ".png"),
            )
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            cv2.imwrite(output_filename, masks)

            # overlay
            output_filename = os.path.join(
                args.target_dir,
                args.split,
                video_id,
                "overlay" + "_" + filename.replace(".jpg", ".png"),
            )
            video_filepaths.append(output_filename)
            plt.imshow(image)
            plt.imshow(masks, alpha=0.5 * masks / 255)
            plt.axis("off")
            plt.savefig(output_filename)
            plt.close()

        video_output_path = os.path.join(
            args.target_dir, args.split, "video", f"{video_id}.mp4"
        )
        os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
        print(f"Generating video {video_output_path}")
        print(f"Generating video {video_id}")
        make_video(video_filepaths, 14, target=video_output_path)


def get_mask_prediction(image, predictor):
    with torch.no_grad():
        with torch.cuda.amp.autocast_mode.autocast():
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=False,
            )
            return masks[0].astype("uint8") * 255


def make_video(image_files, fps, target="video.mp4"):

    im = cv2.imread(image_files[0])
    h, w, c = im.shape

    video = cv2.VideoWriter(
        target, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=True
    )
    for path in tqdm(image_files):
        im = cv2.imread(path)
        video.write(im)

    video.release()


if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    main(args)
