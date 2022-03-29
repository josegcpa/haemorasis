"""
Generate examples of cells in an HDF5 format.

Usage:
    python3 get_examples.py --help
"""

import argparse
import numpy as np
import h5py
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get cell examples.')

    parser.add_argument('--aggregates_path',dest='aggregates_path',
                        action='store',type=str,default=None,
                        help="Path to cell characteristics HDF5")
    parser.add_argument('--segmented_path',dest='segmented_path',
                        action='store',type=str,default=None,
                        help="Path to segmented cells HDF5")
    parser.add_argument('--output_path',dest='output_path',
                        action='store',type=str,default=None,
                        help="Path to output")
    parser.add_argument('--subset',dest='subset',
                        action='store',type=int,default=None,
                        help="No. of cells in each cell collection")

    args = parser.parse_args()

    aggregates = h5py.File(args.aggregates_path,'r')
    segmentations = h5py.File(args.segmented_path,'r')
    output = h5py.File(args.output_path,'w')

    subset_size = args.subset
    all_cell_center_idx = aggregates['cell_center_idxs']
    all_cell_centers = list(segmentations.keys())
    n_cells = len(all_cell_center_idx)
    if subset_size > n_cells:
        subset_size = n_cells
    center_subset = np.random.choice(n_cells,size=subset_size,replace=False)
    center_subset = np.sort(center_subset)
    for cell_idx in tqdm(center_subset):
        center_idx = all_cell_center_idx[cell_idx]
        center = all_cell_centers[center_idx]
        center_int = center[1:-1].split(',')
        center_int = [float(center_int[0]),float(center_int[1])]
        segmented_cell_group = segmentations[center]
        image = segmented_cell_group['image']
        mask = segmented_cell_group['mask']
        cell_center_x = center_int[0]
        cell_center_y = center_int[1]
        features = aggregates['cells']['0'][cell_idx,:]
        g = output.create_group(center)
        g.create_dataset('image',data=image,dtype=np.uint8)
        g.create_dataset('isolated_image',data=mask,dtype=np.uint8)
        g.create_dataset('cell_center_y',data=center_int[0])
        g.create_dataset('cell_center_x',data=center_int[1])
        g.create_dataset('features',data=features)