import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.sam import Sam
import numpy as np
import torch
from torch.nn import functional as F 
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm
import json


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_arg_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser("Finetune SAM to detect ultrasound with no prompt")
    parser.add_argument("--sam_weights", default="sam_vit_b_01ec64.pth", help='Path to SAM weights (download from https://github.com/facebookresearch/segment-anything)')
    parser.add_argument(
        "--sam_type", choices=sam_model_registry.keys(), default="vit_b", help='Which SAM architecture to use'
    )
    parser.add_argument("--data_csv_path", help='Path to the csv describing the train/val dataset')
    parser.add_argument("--masks_dir", help='Path to the masks directory output by `generate_sam_masks_from_boxes.py`')
    parser.add_argument("--splits_file", help='Path to json file containing data splits (e.g. see `splits_trainee.json`)')

    # training
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size for training - SAM is so big that large batch size might not be possible")
    parser.add_argument("--lr", default=1e-5, type=float, help='Learning rate for training')
    parser.add_argument("--use_amp", action="store_true", default=False, help='Use fp16 mixed precision - speeds up training and decreases memory footprint')
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument(
        "--freeze_image_encoder",
        action="store_true",
        help="Whether to freeze the sam image encoder (speed up training)",
    )
    parser.add_argument('--loss', choices=('ce', 'focal'), default='ce',)
    parser.add_argument('--focal_loss_gamma', type=float, help='if using `--loss=focal`, specify the gamma coefficient here.', default=1)

    # misc
    parser.add_argument("--save_dir", help='Path to save checkpoints')
    parser.add_argument("--debug", action="store_true", help='Run in debug mode')


    return parser


def main(args):

    (
        model,
        train_loader,
        val_loader,
        optimizer,
        criteron,
        grad_scaler,
        best_val_score,
        epoch,
    ) = setup_experiment(args)

    wandb.init(project="sam_ultrasound_detection", config=args)

    for epoch in range(epoch, args.epochs):
        pbar = tqdm(train_loader, desc=f"EPOCH {epoch} | Training")
        for i, batch in enumerate(pbar):
            if args.debug and i > 10:
                break
            image, mask = batch
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            with torch.cuda.amp.autocast_mode.autocast(enabled=args.use_amp):
                pred_mask = model(image)
            loss = criteron(pred_mask, mask)

            optimizer.zero_grad()
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            wandb.log({"loss": loss.item()})

        pbar = tqdm(val_loader, desc=f"EPOCH {epoch} | Validation")
        losses = []
        for i, batch in enumerate(pbar):
            if args.debug and i > 10:
                break
            image, mask = batch
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                with torch.cuda.amp.autocast_mode.autocast(enabled=args.use_amp):
                    pred_mask = model(image)
                loss = criteron(pred_mask, mask)
                wandb.log({"val_loss": loss.item()})
                losses.append(loss.item())
        avg_val_loss = sum(losses) / len(losses)
        wandb.log({"val_epoch_loss": avg_val_loss})

        if avg_val_loss < best_val_score:
            print(f"Reached new lowest val loss!")
            print(f"Saving model...")
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"{args.sam_type}_best.pt"),
            )


def setup_experiment(args):
    sam_model = sam_model_registry[args.sam_type](args.sam_weights)
    preprocess_transform = SAMPreprocessingForTraining(sam_model)

    # datasets and loaders
    _make_dataset = lambda split: ImagesAndUltrasoundMasksDataset(
        args.data_csv_path, args.masks_dir, split=split, transform=preprocess_transform, 
        splits_file=args.splits_file
    )
    train_dataset = _make_dataset("train")
    val_dataset = _make_dataset("val")
    _make_loader = lambda is_train, set: DataLoader(
        set,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True,
    )
    train_loader = _make_loader(True, train_dataset)
    val_loader = _make_loader(False, val_dataset)

    # model and optimizer
    model = SamWrapper(sam_model, freeze_backbone=args.freeze_image_encoder).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # criterion
    if args.loss == 'ce': 
        criterion = BinaryCELossFromLogits()
    elif args.loss == 'focal': 
        criterion = BinaryFocalLossFromLogits(gamma=args.focal_loss_gamma)
    else: 
        raise ValueError(f"Unknown loss {args.loss}")

    if args.use_amp:
        grad_scaler = GradScaler()
    else:
        grad_scaler = None

    best_val_score = 1e9
    epoch = 0

    return (
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        grad_scaler,
        best_val_score,
        epoch,
    )


def _fix_gt_size(pred_mask, gt_mask): 
    h, w = pred_mask.shape[-2:]
    gt_mask = torch.nn.functional.interpolate(gt_mask.float()[:, None, ...], (h, w))
    return gt_mask


class BinaryCELossFromLogits:
    def __call__(self, pred_mask, gt_mask):
        # pred mask is low res, need to resize gt
        gt_mask = _fix_gt_size(pred_mask, gt_mask)
        return torch.nn.functional.binary_cross_entropy_with_logits(pred_mask, gt_mask)


class BinaryFocalLossFromLogits: 
    def __init__(self, gamma=1): 
        self.gamma = gamma

    def __call__(self, pred_mask, gt_mask): 
        gt_mask = _fix_gt_size(pred_mask, gt_mask)
        
        pred_mask = pred_mask.sigmoid()

        y_hat_pos = (pred_mask)
        y_hat_neg = (1 - pred_mask)

        pos_term = y_hat_pos.log() * (1 - y_hat_pos) ** self.gamma
        neg_term = y_hat_neg.log() * (1 - y_hat_neg) ** self.gamma 

        unreduced_loss = gt_mask * pos_term + (1 - gt_mask) * neg_term
        unreduced_loss = -unreduced_loss

        return unreduced_loss.mean(1)




class ImagesAndUltrasoundMasksDataset:
    def __init__(
        self,
        data_csv_path,
        masks_dir=None,
        split="train",
        transform=None,
        splits_file="splits_trainee.json",
    ):
        with open(splits_file) as f:
            splits = json.load(f)

        self.data_table = pd.read_csv(data_csv_path, index_col=0)
        self.data_table["VideoID"] = self.data_table.FileName.apply(
            lambda x: x.split("_")[0]
        )
        self.data_table["TraineeID"] = self.data_table.FileName.apply(
            lambda x: x.split("-")[0]
        )
        splits_ids = splits[split]
        print(f"Using IDS: \n{splits_ids}\n for split {split}")
        self.data_table = self.data_table.loc[
            self.data_table["TraineeID"].isin(splits_ids)
        ]
        print(f"Number of videos: {len(self.data_table['VideoID'].unique())}")
        print(f"Number of frames: {len(self)}")
        self.transform = transform
        self.masks_dir = masks_dir

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, idx):
        if self.masks_dir is None:
            raise ValueError("To use getitem, you need to specify `masks_dir`")
        row = self.data_table.iloc[idx].to_dict()

        # get path to image
        filename = row["FileName"]
        folder = row["Folder"]
        fullpath = os.path.join(folder, filename)

        image = Image.open(fullpath)

        mask_prefix = "ultrasound_"
        mask_path = mask_prefix + filename.replace("jpg", "png")
        fullpath = os.path.join(self.masks_dir, mask_path)

        mask = Image.open(fullpath)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


class SAMPreprocessingForTraining:
    def __init__(self, sam_model: Sam):
        self.resize_longest_side = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.img_size = sam_model.image_encoder.img_size
        self.pixel_mean = sam_model.pixel_mean
        self.pixel_std = sam_model.pixel_std

    def __call__(self, image, mask):
        image = np.array(image)
        mask = np.array(mask)

        image = self.resize_longest_side.apply_image(image)
        mask = self.resize_longest_side.apply_image(mask)
        image_torch = torch.as_tensor(image)
        image_torch = image_torch.permute(2, 0, 1).contiguous()[None, ...]
        mask_torch = torch.as_tensor(mask)
        mask_torch = mask_torch.contiguous()[None, ...]

        image_torch = (image_torch - self.pixel_mean) / self.pixel_std

        # pad
        h, w = image_torch.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        image_torch = torch.nn.functional.pad(image_torch, (0, padw, 0, padh))[0]
        mask_torch = torch.nn.functional.pad(mask_torch, (0, padw, 0, padh))[0]
        mask_torch = (mask_torch / 255).long()

        return image_torch, mask_torch


class SamWrapper(torch.nn.Module):

    def __init__(self, sam_model: Sam, freeze_backbone=False):
        super().__init__()
        self.model = sam_model
        self.freeze_backbone = freeze_backbone

    def forward(self, image):
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            features = self.model.image_encoder(image)

        # get prompt embeddings with no prompt
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            None, None, None
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return low_res_masks


if __name__ == "__main__":
    parser = get_arg_parser()
    main(parser.parse_args())
