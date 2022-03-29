"""
Generate images from cell examples.

Usage:
    python3 get_images_from_examples.py --help
"""

import h5py
import numpy as np
from PIL import Image
import os
import argparse
from glob import glob
from tqdm import tqdm

try:
    os.makedirs("images_from_examples")
except:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate images from cell examples.')

    parser.add_argument('--input_path',dest='input_path',
                        action='store',type=str,default=None,
                        help="Path to folder with cell examples")
    parser.add_argument('--cell_type',dest='cell_type',
                        action='store',type=str,default=None,
                        help="Type of cell (rbc or wbc, used as pattern)")
    parser.add_argument('--output_path',dest='output_path',
                        action='store',type=str,default=None,
                        help="Output path for images")

    args = parser.parse_args()

    all_examples = glob(os.path.join(args.input_path,"*" + args.cell_type + "*.h5"))

    try:
        os.makedirs(args.output_path)
    except:
        pass

    N = 20
    H,W = 250,130

    all_isolated_images_mll = []

    for file_path in tqdm(all_examples):
        
        with h5py.File(file_path,"r") as F:
            root = file_path.split(os.sep)[-1]

            patch_work = np.zeros([H*N,W*N,3],dtype=np.uint8)

            for i,k in enumerate(F):
                image = F[k]['image'][::]
                seg_image = F[k]['isolated_image'][::]
                isolated_image = np.where(
                    seg_image[:,:,np.newaxis] > 0,image,np.uint8(image * 0.5)
                )
                x,y = np.where(seg_image > 0)

                isolated_image = np.concatenate([image,isolated_image])
                sh = isolated_image.shape

                all_isolated_images_mll.append((isolated_image,F,isolated_image.shape))

                if i < (N**2):
                    place_x,place_y = int(i // N),(i % N)
                    place_x_,place_y_ = place_x*H,place_y*W
                    place_x = int(place_x_ + int(H/2 - sh[0] / 2))
                    place_y = int(place_y_ + int(W/2 - sh[1] / 2))
                    try:
                        patch_work[place_x:(place_x+sh[0]),place_y:(place_y+sh[1])] = isolated_image
                    except:
                        pass

        output_image_path = os.path.join(args.output_path,'{}_{}.png'.format(args.cell_type,root))
        if os.path.exists(output_image_path) == False:
            patch_work = patch_work[:(place_x_ + H),:]
            Image.fromarray(patch_work).save(output_image_path)

    N = 25

    np.random.seed(4422)

    isolated_image_idx = np.random.choice(len(all_isolated_images_mll),N*N,replace=False)

    isolated_images = [all_isolated_images_mll[i] 
                    for i in isolated_image_idx]

    H,W = [np.max([x[2][0] for x in isolated_images])+2,
        np.max([x[2][1] for x in isolated_images])+2]

    patch_work = np.zeros([H*N,W*N,3],dtype=np.uint8)

    for i,k in enumerate(isolated_images):
        isolated_image = k[0]
        sh = isolated_image.shape
        place_x,place_y = int(i // N),(i % N)
        place_x_,place_y_ = place_x*H,place_y*W
        place_x = int(place_x_ + int(H/2 - sh[0] / 2))
        place_y = int(place_y_ + int(W/2 - sh[1] / 2))

        patch_work[place_x:(place_x+sh[0]),place_y:(place_y+sh[1])] = isolated_image

    Image.fromarray(patch_work).save(
        os.path.join(args.output_path,'{}.png'.format(args.cell_type)))