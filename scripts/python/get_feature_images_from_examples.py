"""
Generates example images of the morphometric feature distribution.
Each feature is described by a given number of quantiles, with
images corresponding to each quantile from lowest (left) to
highest (right)

Usage:
    python3 get_feature_images_from_examples.py --help
"""

import argparse
import h5py
import numpy as np
from PIL import Image
import os
import re
from glob import glob
from tqdm import tqdm 

from MIA import MIA_FEATURES,MIA_FEATURES_NUCLEUS

all_feature_names = MIA_FEATURES
all_feature_names.extend([x+"_nuclear" for x in MIA_FEATURES_NUCLEUS])

def pad_to_size(image,size):
    h,w,_ = image.shape
    h_a = (size[0] - h) // 2
    w_a = (size[1] - w) // 2
    h_b = (size[0] - h_a - h)
    w_b = (size[1] - w_a - w)
    image = np.pad(image,((h_a, h_b), (w_a, w_b), (0, 0)),
                   mode='constant',constant_values=0)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get cell feature examples.')

    parser.add_argument('--dataset',dest='dataset',
                        action='store',type=str,choices=["MLL","AC2"],
                        default="MLL",help="Dataset ID")
    parser.add_argument('--cell_type',dest='cell_type',
                        action='store',type=str,choices=["WBC","RBC"],
                        default="WBC",help="Cell type")
    parser.add_argument('--N_examples',dest='N_examples',
                        action='store',type=int,default=10,
                        help="No. of cells per quantile")
    parser.add_argument('--N_quantiles',dest='N_quantiles',
                        action='store',type=int,default=10,
                        help="Number of sampled quantiles")

    args = parser.parse_args()

    try:
        os.makedirs("feature_images_from_examples")
    except:
        pass

    if args.cell_type == "WBC":
        all_examples = glob("examples/*wbc.h5")
    if args.cell_type == "RBC":
        all_examples = glob("examples/*rbc.h5")

    np.random.shuffle(all_examples)

    if args.dataset == "MLL":
        examples = [x for x in all_examples if re.search('.*[A-Z]+_[0-9]+_[wr]bc',x)]
    elif args.dataset == "AC2":
        examples = [x for x in all_examples if not re.search('.*[A-Z]+_[0-9]+_[wr]bc',x)]

    # get quantiles
    print("Retrieving quantiles...")
    all_features = []
    for file_path in tqdm(examples):
        with h5py.File(file_path,"r") as F:
            root = file_path.split(os.sep)[-1]
            centers = []
            hists = []

            for i,k in enumerate(F):
                features = F[k]['features'][::]
                all_features.append(features)

    all_features = np.stack(all_features,axis=0)
    quantiles = np.quantile(all_features,
                            [i/args.N_quantiles for i in range(1,args.N_quantiles+1)],axis=0)

    N = quantiles.shape[0]
    N_examples = args.N_examples
    H,W = 210,105

    all_idxs = [[[] for _ in range(quantiles.shape[0])] 
                for _ in range(quantiles.shape[1])]
    all_isolated_images = []

    image_idx = 0

    print("Collecting images...")
    for file_path in tqdm(examples):
        with h5py.File(file_path,"r") as F:
            root = file_path.split(os.sep)[-1]

            all_keys = [x for x in F.keys()]
            np.random.shuffle(all_keys)

            for i,k in enumerate(all_keys):
                image = F[k]['image'][::]
                seg_image = F[k]['isolated_image'][::]
                features = F[k]['features'][()]
                q_idx = np.argmin(features > quantiles,axis=0)

                isolated_image = np.where(
                    seg_image[:,:,np.newaxis] > 0,image,np.uint8(image * 0.5)
                )

                isolated_image = np.concatenate([image,isolated_image])

                all_isolated_images.append(isolated_image)
                for j,q in enumerate(q_idx):
                    all_idxs[j][q].append(image_idx)
                image_idx += 1

    print("Writing images...")
    for i,quantile_idx in enumerate(tqdm(all_idxs)):
        patch_work = np.zeros([H*N_examples,W*N,4],dtype=np.uint8)
        for j in range(quantiles.shape[0]):
            idxs = all_idxs[i][j]
            if len(idxs) < N_examples:
                idx_sample = idxs
            else:
                idx_sample = np.random.choice(idxs,size=N_examples,replace=False)
            images = []
            for idx in idx_sample:
                images.append(pad_to_size(all_isolated_images[idx],(H,W)))
            for _ in range(N_examples-len(images)):
                images.append(np.zeros((H,W,3)))
            
            images = np.concatenate(images)
            patch_work[:,(j*W):(j*W+W),:3] = images
            patch_work[:,(j*W):(j*W+W),3] = 255

            output_image_path = "feature_images_from_examples/{}_{}_{}_{}quant.png".format(
                args.dataset,args.cell_type,all_feature_names[i],args.N_quantiles)
            Image.fromarray(patch_work).save(output_image_path)