"""
Gets the tile colour histograms of a histopathology slide.

Usage:
    python3 get_slide_colour_histograms.py --help
"""

import argparse
import numpy as np

from image_generator import image_generator_slide

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get the colour histogram of all slide tiles.')

    parser.add_argument('--slide_path',dest='slide_path',
                        action='store',type=str,default=None,
                        help="Path to slide")
    args = parser.parse_args()

    for tile,coords in image_generator_slide(args.slide_path,1024,1024):
        r_hist,_ = np.histogram(
            tile[:,:,0],bins=255,range=[0,255],normed=False)
        g_hist,_ = np.histogram(
            tile[:,:,1],bins=255,range=[0,255],normed=False)
        b_hist,_ = np.histogram(
            tile[:,:,2],bins=255,range=[0,255],normed=False)

        av_hist,_ = np.histogram(
            np.mean(tile,axis=-1),bins=255,range=[0,255],normed=False)

        print('HIST,{},{},{}'.format(
            coords,'r',','.join([str(x) for x in r_hist.tolist()])))
        print('HIST,{},{},{}'.format(
            coords,'g',','.join([str(x) for x in g_hist.tolist()])))
        print('HIST,{},{},{}'.format(
            coords,'b',','.join([str(x) for x in b_hist.tolist()])))
        print('HIST,{},{},{}'.format(
            coords,'av',','.join([str(x) for x in av_hist.tolist()])))
