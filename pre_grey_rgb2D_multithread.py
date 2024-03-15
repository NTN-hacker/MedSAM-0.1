import argparse
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import numpy as np
import os
from glob import glob
import pandas as pd
import cv2

join = os.path.join
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse

def main(args):

    opt = {}
    opt['n_thread'] = args.n_thread
    opt['compression_level'] = args.compression_level
    opt['img_path'] = args.img_path
    opt['gt_path'] = args.gt_path
    opt['data_name'] = args.data_name
    opt['npz_path'] = args.npz_path
    opt['image_size'] = args.image_size
    opt['img_name_suffix'] = args.img_name_suffix
    opt['label_id'] = args.label_id
    opt['model_type'] = args.model_type
    opt['checkpoint'] = args.checkpoint
    opt['device'] = args.device
    opt['seed'] = args.seed
    opt['thresh_size'] = args.thresh_size
    convert(opt)


def convert(opt):
    # convert 2d grey or rgb images to npz file

    sam_model = sam_model_registry[opt['model_type']](checkpoint=args.checkpoint).to(
        opt['device']
    )

    names = sorted(os.listdir(opt['gt_path']))
    # print the number of images found in the ground truth folder
    print("image number:", len(names))

    # print(names)
    pbar = tqdm(total=len(names), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for gt_name in names:
        pool.apply_async(worker, args=(gt_name, None, opt, sam_model), callback=lambda arg: pbar.update(1))

    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')

    # create a directory to save the npz files
    save_path = opt['npz_path'] + "_" + opt['model_type']
    os.makedirs(save_path, exist_ok=True)
    print("Num. of images:", imgs)
    print('check', len(imgs))

    if len(imgs) > 1:
        imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
        gts = np.stack(gts, axis=0)  # (n, 256, 256)
        img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)

        np.savez_compressed(
            join(save_path, opt['data_name'] + ".npz"),
            imgs=imgs,
            gts=gts,
            img_embeddings=img_embeddings,

        )
        # save an example image for sanity check
        idx = np.random.randint(imgs.shape[0])
        img_idx = imgs[idx, :, :, :]
        gt_idx = gts[idx, :, :]
        bd = segmentation.find_boundaries(gt_idx, mode="inner")
        img_idx[bd, :] = [255, 0, 0]
        io.imsave(save_path + ".png", img_idx, check_contrast=False)
    else:
        print(
            "Do not find image and ground-truth pairs. Please check your dataset and argument settings"
        )



def worker(gt_name, image_name, opt, sam_model):
    print(1)

    if image_name == None:
        image_name = gt_name
        print(image_name)
    # gt_data = io.imread(join(args.gt_path, gt_name), as_gray=True)
    gt_data = cv2.imread(join(opt['gt_path'], gt_name), 0)
    # if it is rgb, select the first channel
    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    assert len(gt_data.shape) == 2, "ground truth should be 2D"

    # # # resize ground truch image
    gt_data = transform.resize(
        gt_data,
        (opt['image_size'], opt['image_size']),
        order=0,
        preserve_range=True,
        mode="constant",
        anti_aliasing=True
    )

    (thresh, gt_data) = cv2.threshold(gt_data, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gt_data = gt_data.astype(np.uint8) / 255
    # convert to uint8
    gt_data = np.uint8(gt_data)
    print(np.sum(gt_data))
    if np.sum(gt_data) > 40:  # exclude tiny objects
        """Optional binary thresholding can be added"""
        assert (
                np.max(gt_data) == 1 and np.unique(gt_data).shape[0] == 2
        ), "ground truth should be binary"

        image_data = io.imread(join(opt['img_path'], image_name))
        # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(
            image_data, 99.5
        )
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        # min-max normalize and scale
        image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
        )
        image_data_pre[image_data == 0] = 0

        image_data_pre = transform.resize(
            image_data_pre,
            (opt['image_size'], opt['image_size']),
            order=3,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True,
        )
        image_data_pre = np.uint8(image_data_pre)
        # print(image_data_pre.shape)
        print(gt_data.shape)

        assert np.sum(gt_data) > 40, "ground truth should have more than 50 pixels"

        # imgs.append(image_data_pre)
        # print('check imgs', imgs.shape)
        # gts.append(gt_data)
        # resize image to 3*1024*1024
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image_data_pre)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(
            opt['device']
        )
        input_image = sam_model.preprocess(
            resize_img_tensor[None, :, :, :]
        )  # (1, 3, 1024, 1024)
        assert input_image.shape == (
            1,
            3,
            sam_model.image_encoder.img_size,
            sam_model.image_encoder.img_size,
        ), "input image should be resized to 1024*1024"
        # pre-compute the image embedding
        with torch.no_grad():
            print(input_image.shape)
            embedding = sam_model.image_encoder(input_image)
            img_embedding = embedding.cpu().numpy()[0]
            # img_embeddings.append(img_embedding)
        
        print('check image_data_pre', image_data_pre)

    # return image_data_pre, gt_data, img_embedding

# def worker(path, opt):
#     """Worker for each process.

#     Args:
#         path (str): Image path.
#         opt (dict): Configuration dict. It contains:
#             crop_size (int): Crop size.
#             step (int): Step for overlapped sliding window.
#             thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.
#             save_folder (str): Path to save folder.
#             compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

#     Returns:
#         process_info (str): Process information displayed in progress bar.
#     """
#     crop_size = opt['crop_size']
#     step = opt['step']
#     thresh_size = opt['thresh_size']
#     img_name, extension = osp.splitext(osp.basename(path))

#     # remove the x2, x3, x4 and x8 in the filename for DIV2K
#     img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

#     h, w = img.shape[0:2]
#     h_space = np.arange(0, h - crop_size + 1, step)
#     if h - (h_space[-1] + crop_size) > thresh_size:
#         h_space = np.append(h_space, h - crop_size)
#     w_space = np.arange(0, w - crop_size + 1, step)
#     if w - (w_space[-1] + crop_size) > thresh_size:
#         w_space = np.append(w_space, w - crop_size)

#     index = 0
#     for x in h_space:
#         for y in w_space:
#             index += 1
#             cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
#             cropped_img = np.ascontiguousarray(cropped_img)
#             cv2.imwrite(
#                 osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
#                 [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
#     process_info = f'Processing {img_name} ...'
#     return process_info


if __name__ == '__main__':
    # set up the parser
    parser = argparse.ArgumentParser(description="preprocess grey and RGB images")

    # add arguments to the parser
    parser.add_argument(
        "-i",
        "--img_path",
        type=str,
        default="data/sam/augmentation/Transistor/images",
        help="path to the images",
    )
    parser.add_argument(
        "-gt",
        "--gt_path",
        type=str,
        default="data/sam/augmentation/Transistor/labels",
        help="path to the ground truth (gt)",
    )

    parser.add_argument(
        "-o",
        "--npz_path",
        type=str,
        default="data/demo2D",
        help="path to save the npz files",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="2024_03_15/Transistor/transistor",
        help="dataset name; used to name the final npz file, e.g., demo2d.npz",
    )
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument(
        "--img_name_suffix", type=str, default=".png", help="image name suffix"
    )
    parser.add_argument("--label_id", type=int, default=255, help="label id")
    parser.add_argument("--model_type", type=str, default="vit_b", help="model type")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="work_dir/SAM/sam_vit_b_01ec64.pth",
        help="checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")

    parser.add_argument(
        '--thresh_size',
        type=int,
        default=0,
        help='Threshold size. Patches whose size is lower than thresh_size will be dropped.')
    parser.add_argument('--n_thread', type=int, default=20, help='Thread number.')
    parser.add_argument('--compression_level', type=int, default=3, help='Compression level')
    args = parser.parse_args()

    main(args)