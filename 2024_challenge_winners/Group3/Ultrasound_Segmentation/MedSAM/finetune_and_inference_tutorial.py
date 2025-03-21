# %% set up environment
import time
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient, hausdorff_distance, extract_points
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)





start_time = time.time()




#%% create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['embeddings'] for d in self.npz_data])
        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        y_indices, x_indices = np.where(gt2D > 0)
        y_min = y_max = x_min = x_max = 0  # Default values
        if y_indices.shape[0] > 0:
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        if x_indices.shape[0] > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)

        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()


# %% test dataset class and dataloader
npz_tr_path = 'data/Npz_files/our_data/train'
demo_dataset = NpzDataset(npz_tr_path)
demo_dataloader = DataLoader(demo_dataset, batch_size=8, shuffle=True)
for img_embed, gt2D, bboxes in demo_dataloader:
    # img_embed: (B, 256, 64, 64), gt2D: (B, 1, 256, 256), bboxes: (B, 4)
    print(f"{img_embed.shape=}, {gt2D.shape=}, {bboxes.shape=}")
    break

# %% set up model for fine-tuning 
# train data path
npz_tr_path = 'data/Npz_files/our_data/train'
work_dir = os.getcwd()

# prepare SAM model
model_type = 'vit_b'
checkpoint = 'SAM/sam_vit_b_01ec64.pth'
device = 'cuda:0'

model_save_path = work_dir
os.makedirs(model_save_path, exist_ok=True)

sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
sam_model.train()


optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# regress loss for IoU/DSC prediction; (ignored for simplicity but will definitely included in the near future)
# regress_loss = torch.nn.MSELoss(reduction='mean')


#%% train
num_epochs = 100
losses = []
best_loss = 1e10
train_dataset = NpzDataset(npz_tr_path)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for epoch in range(num_epochs):
    epoch_loss = 0
    # train
    for step, (image_embedding, gt2D, boxes) in enumerate(tqdm(train_dataloader)):
        # do not compute gradients for image encoder and prompt encoder
        with torch.no_grad():
            # convert box to 1024x1024 grid
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)
            # get prompt embeddings 
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
        # predicted masks
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          )

        loss = seg_loss(mask_predictions, gt2D.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss /= step
    losses.append(epoch_loss)
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
    # save the latest model checkpoint
    torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_latest.pth'))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best_balanced_3000.pth'))

    # plot loss
    plt.plot(losses)
    plt.title('Dice + Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show() # comment this line if you are running on a server
    plt.savefig(join(model_save_path, 'train_loss.png'))
    plt.close()
    
# torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best_balanced_3000.pth'))    
    
    
    
end_time = time.time()

# Calculate the elapsed time
execution_time = end_time - start_time

# Print the execution time
print("Execution time:", execution_time, "seconds")

#%% compare the segmentation results between the original SAM model and the fine-tuned model
# load the original SAM model
ori_sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
ori_sam_predictor = SamPredictor(ori_sam_model)
npz_ts_path = 'data/Npz_files/our_data/test'
test_npzs = sorted(os.listdir(npz_ts_path))
# random select a test case
npz_idx = np.random.randint(0, len(test_npzs))
npz = np.load(join(npz_ts_path, test_npzs[npz_idx]))
imgs = npz['imgs']
gts = npz['gts']

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
for img, gt in zip(imgs, gts):
    bbox = get_bbox_from_mask(gt)
    bboxes.append(bbox)
    # predict the segmentation mask using the original SAM model
    ori_sam_predictor.set_image(img)
    ori_sam_seg, _, _ = ori_sam_predictor.predict(point_coords=None, box=None, multimask_output=False)
    ori_sam_segs.append(ori_sam_seg[0])
    
    # predict the segmentation mask using the fine-tuned model
    H, W = img.shape[:2]
    resize_img = sam_trans.apply_image(img)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        bbox = sam_trans.apply_boxes(bbox, (H, W))
        box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        medsam_segs.append(medsam_seg)

#%% compute the DSC score
ori_sam_segs = np.stack(ori_sam_segs, axis=0)
medsam_segs = np.stack(medsam_segs, axis=0)

ori_sam_dsc = compute_dice_coefficient(gts>0, ori_sam_segs>0)
medsam_dsc = compute_dice_coefficient(gts>0, medsam_segs>0)
print('Original SAM DSC: {:.4f}'.format(ori_sam_dsc), 'MedSAM DSC: {:.4f}'.format(medsam_dsc))

ori_sam_dsc_list = []
medsam_dsc_list = []
for ind_img in range(len(gts)):
    ori_sam_dsc_ind = compute_dice_coefficient(gts[ind_img]>0, ori_sam_segs[ind_img]>0)
    medsam_dsc_ind = compute_dice_coefficient(gts[ind_img]>0, medsam_segs[ind_img]>0)
    
    ori_sam_dsc_list.append(ori_sam_dsc_ind)
    medsam_dsc_list.append(medsam_dsc_ind)
    
print('MEAN_Original SAM DSC: {:.4f}'.format(np.nanmean(ori_sam_dsc_list)), 'MEAN_MedSAM DSC: {:.4f}'.format(np.nanmean(medsam_dsc_list)))
print('STD_Original SAM DSC: {:.4f}'.format(np.nanstd(ori_sam_dsc_list)), 'STD_MedSAM DSC: {:.4f}'.format(np.nanstd(medsam_dsc_list)))
#%% compute the hausdorph distance

# Extract the boundary points from the binary masks
gts_points = extract_points(gts > 0)
ori_sam_points = extract_points(ori_sam_segs)
medsam_points = extract_points(medsam_segs > 0)

# Calculate Hausdorff distances
ori_sam_hd = hausdorff_distance(gts_points, ori_sam_points)
medsam_hd = hausdorff_distance(gts_points, medsam_points)

print("Hausdorff Distance between gts and ori_sam_segs:", ori_sam_hd)
print("Hausdorff Distance between gts and medsam_segs:", medsam_hd)



ori_sam_hd_list = []
medsam_hd_list = []
for ind_img in range(len(gts)):
    gts_points = extract_points(gts[ind_img] > 0)
    ori_sam_points = extract_points(ori_sam_segs[ind_img])
    medsam_points = extract_points(medsam_segs[ind_img] > 0)

    # Calculate Hausdorff distances
    ori_sam_hd = hausdorff_distance(gts_points, ori_sam_points)
    medsam_hd = hausdorff_distance(gts_points, medsam_points)
    
    ori_sam_hd_list.append(ori_sam_hd)
    medsam_hd_list.append(medsam_hd)
    
print('MEAN_Original SAM HD: {:.4f}'.format(np.mean(ori_sam_hd_list)), 'MEAN_MedSAM DSC: {:.4f}'.format(np.mean(medsam_hd_list)))
print('STD_Original SAM HD: {:.4f}'.format(np.std(ori_sam_hd_list)), 'STD_MedSAM DSC: {:.4f}'.format(np.std(medsam_hd_list)))


#%% visualize the segmentation results of the middle slice
# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))    


low_score_indices = [index for index, score in enumerate(medsam_dsc_list) if score < 0.5]
for iind in low_score_indices:
    
    img_id = iind
    
    _, axs = plt.subplots(1, 3, figsize=(25, 25))
    axs[0].imshow(imgs[img_id])
    show_mask(gts[img_id], axs[0])
    # show_box(box_np[img_id], axs[0])
    # axs[0].set_title('Mask with Tuned Model', fontsize=20)
    axs[0].axis('off')
    
    axs[1].imshow(imgs[img_id])
    show_mask(ori_sam_segs[img_id], axs[1])
    show_box(bboxes[img_id], axs[1])
    # add text to image to show dice score
    axs[1].text(0.5, 0.5, 'SAM DSC: {:.4f}'.format(ori_sam_dsc), fontsize=30, horizontalalignment='left', verticalalignment='top', color='yellow')
    # axs[1].set_title('Mask with Untuned Model', fontsize=20)
    axs[1].axis('off')
    
    axs[2].imshow(imgs[img_id])
    show_mask(medsam_segs[img_id], axs[2])
    show_box(bboxes[img_id], axs[2])
    # add text to image to show dice score
    axs[2].text(0.5, 0.5, 'MedSAM DSC: {:.4f}'.format(medsam_dsc), fontsize=30, horizontalalignment='left', verticalalignment='top', color='yellow')
    # axs[2].set_title('Ground Truth', fontsize=20)
    axs[2].axis('off')
    plt.show()  
    plt.subplots_adjust(wspace=0.01, hspace=0)
    # save plot
    # plt.savefig(join(model_save_path, test_npzs[npz_idx].split('.npz')[0] + str(img_id).zfill(3) + '.png'), bbox_inches='tight', dpi=300)
    plt.close()
