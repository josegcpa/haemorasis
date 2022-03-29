"""
Generates a slide thumbnail from a histopathology slide.

Usage:
    python3 get_slide_thumbnail.py --help
"""

import argparse
import numpy as np
from skimage import filters
from tqdm import tqdm
from PIL import Image

from image_generator import image_generator_slide

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get a smaller version of a slide.')

    parser.add_argument('--slide_path',dest='slide_path',
                        action='store',type=str,default=None,
                        help="Path to slide")
    parser.add_argument('--output_path',dest='output_path',
                        action='store',type=str,default=None,
                        help="Path to output thumbnail")
    args = parser.parse_args()

    sigma = 3
    d = 64
    s = 1024

    all_x = []
    all_y = []
    all_tiles = []
    for tile,coords in tqdm(image_generator_slide(args.slide_path,s,s)):
        y,x = [int(x)//d for x in coords.split(',')]
        tile = filters.gaussian(
            tile, sigma=(sigma,sigma),multichannel=True)[::d,::d,]
        tile = np.uint8(tile*255)
        all_tiles.append(tile)
        all_x.append(x)
        all_y.append(y)

    max_x = np.max(all_x)
    max_y = np.max(all_y)

    tos = s//d# tile output size
    output_image = np.zeros([max_x+tos,max_y+tos,3],dtype=np.uint8)
    for x,y,t in zip(all_x,all_y,all_tiles):
        output_image[x:(x+tos),y:(y+tos),:] = t
    
    Image.fromarray(output_image).save(args.output_path)