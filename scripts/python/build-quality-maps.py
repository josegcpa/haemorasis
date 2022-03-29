"""
Aggregates quality control CSV files to an HDF5 containing quality maps
(i.e. 2-dimensional arrays where each cell corrersponds to a tile in 
the figure and its corresponding probability of being of good quality)

Usage:
    python3 build-quality-maps.py --help
"""

import os
import argparse
import numpy as np
import h5py
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'build-quality-maps.py',
        description = 'Aggregates quality control CSVs as quality arrays')

    parser.add_argument('--input_path',dest='input_path',
                        action='store',default=None,
                        help='Path to folder with QC files.')
    parser.add_argument('--output_path',dest='output_path',
                        action='store',default=None,
                        help='Path to output file (hdf5).')

    args = parser.parse_args()

    with h5py.File(args.output_path,'w') as F:
        for file_path in tqdm(glob(os.path.join(args.input_path,'*'))):
            R = os.path.split(file_path)[-1]
            with open(file_path,'r') as o:
                lines = o.readlines()
            lines = [x.strip() for x in lines]
            all_coords = []
            probabilities = []
            for line in lines:
                if line[:3] == "OUT":
                    line = line.split(',')
                    all_coords.append([int(line[1]),int(line[2])])
                    probabilities.append(float(line[-1]))

            all_coords = np.floor(np.array(all_coords) / 512).astype(
                np.int16)
            if all_coords.size > 0:
                quality_map = np.zeros(np.max(all_coords,axis=0)+1)
                quality_map[all_coords[:,0],all_coords[:,1]] = probabilities
                quality_map = quality_map.astype(np.float32)
                F[R] = quality_map
