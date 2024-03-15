#code 


"""
The file to finetune for another samples dataset.
With our parameters: 
- epoch: 10000
- batch_size: 64
- format_input_images: .npz
- image_size: 10242x1024
- run: netron-48
"""


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
from utils.SurfaceDice import compute_dice_coefficient

import neptune
import torch
torch.cuda.empty_cache()

import gc
gc.collect()

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)
type = 'transistor'

##%% create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")

    def __len__(self):
        return self.ori_gts.shape[0]
    
    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]        
        
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()


def main():
    run = neptune.init_run(
        project="nguyennhan8521/medsam",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3NTdkNGM5OS04MjViLTRhMzgtOWNkMy02N2Y3MDI5YjgxZDQifQ==",
    ) 


    # %% set up model for fine-tuning 
    # train data path
    npz_tr_path = 'data/demo2D_vit_b/2024_03_15/Transistor'
    work_dir = './work_dir'
    task_name = 'demo2D'
    # prepare SAM model
    model_type = 'vit_b'
    checkpoint = 'work_dir/SAM/sam_vit_b_01ec64.pth'
    device = 'cuda:0'
    model_save_path = join(work_dir, task_name)
    os.makedirs(model_save_path, exist_ok=True)
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    sam_model.train()
    # Set up the optimizer, hyperparameter tuning will improve performance here
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    total_params = sum(p.numel() for p in sam_model.mask_decoder.parameters())
    print("Total number of parameters in the mask_decoder:", total_params)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    run['information'] = {
        'model_type' : 'vit_b',
        'checkpoint' : 'work_dir/SAM/sam_vit_b_01ec64.pth',
        'device' : 'cuda:0',
        "encoder_parameters": total_params
    }


    ##%% train
    num_epochs = 10000
    batch_size = 64
    losses = []
    best_loss = 1e10
    train_dataset = NpzDataset(npz_tr_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers= 0, pin_memory= True)
    run["parameters"] = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "optimizer": "adam"
    }

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
                    boxes=box_torch,
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
        run["train/epoch/loss"].append(epoch_loss)
        # save the latest model checkpoint
        torch.save(sam_model.state_dict(), join(model_save_path, f'sam_model_latest_{type}.pth'))
        # run["model/weights"].upload(sam_model.state_dict())
        # save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(sam_model.state_dict(), join(model_save_path, f'sam_model_best_{type}.pth'))


    # plot loss
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    plt.plot(losses)
    plt.title('Dice + Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show() # comment this line if you are running on a server
    fig.savefig(join(model_save_path, f'train_loss_{type}.png'))
    plt.close(fig)

    # You can also upload plot objects directly
    run["train/loss"].upload(join(model_save_path, f'train_loss_{type}.png'))

    run.stop()

if __name__ == '__main__':
    main()

