"""
Creates geojson file with annotations from HDF5 files with 
characteristics and cell segmentations

Usage:
    python3 cells-to-annotations.py --help
"""

import h5py
import argparse
import numpy as np
from tqdm import tqdm

def make_dict(x1,x2,y1,y2,colour,name):
    coordinates = [
        [x1,y1],[x1,y2],[x2,y2],[x2,y1],[x1,y1]
    ]
    return {'type': 'Feature',
        'geometry': {'type': 'Polygon', 'coordinates': [coordinates]},
        'properties': {'color': colour, 'isLocked': False,
                       'name': name, 'object_type': 'annotation'}}

edge = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'cells-to-annotations.py',
        description = 'Creates geojson file with annotations')

    parser.add_argument('--segmented_cells_path',dest='segmented_cells_path',
                        action='store',default=None,
                        help='Path to hdf5 file containing segmented cells.')
    parser.add_argument('--characterized_cells_path',dest='characterized_cells_path',
                        action='store',default=None,
                        help='Path to hdf5 file containing characterized cells.')
    parser.add_argument('--name',dest='name',
                        action='store',default=None,
                        help='name of objects in geojson')
    parser.add_argument('--colour',dest='colour',
                        action='store',default=None,nargs='+',
                        help='colour of objects (format "r g b")')
    parser.add_argument('--output_geojson',dest='output_geojson',
                        action='store',default=None,
                        help='output path for geojson file')

    args = parser.parse_args()

    colour = args.colour

    with h5py.File(args.characterized_cells_path,'r') as F:
        cell_center_idxs = F['cell_center_idxs'][()]
    out = open(args.output_geojson,'w')
    out.write('[\n')
    with h5py.File(args.segmented_cells_path,'r') as F:
        all_keys = list(F.keys())
        all_keys = [all_keys[i] for i in cell_center_idxs]
        for k in tqdm(all_keys):
            x = F[k]['X']
            y = F[k]['Y']
            x1,x2,y1,y2 = np.min(x)-edge,np.max(x)+edge,np.min(y)-edge,np.max(y)+edge
            ann = make_dict(x1,x2,y1,y2,colour,args.name)
            out.write('\t' + str(ann) + ',\n')
    out.write(']\n')
    out.close()
