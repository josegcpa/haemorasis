"""
Predicts which tiles are of good quality in WBS.

Usage:
    python3 quality_control.py --help
"""

import argparse
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from glob import glob

from quality_net_utilities import *
from image_generator import *

n_channels = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predicts which tiles are of good quality in WBS.')

    parser.add_argument('--slide_path',dest='slide_path',
                        action='store',type=str,default=None,
                        help="Path to slide.")
    parser.add_argument('--input_height',dest = 'input_height',
                        action = 'store',type = int,default = 512,
                        help = 'The file extension for all images.')
    parser.add_argument('--input_width',dest = 'input_width',
                        action = 'store',type = int,default = 512,
                        help = 'The file extension for all images.')
    parser.add_argument('--checkpoint_path',dest = 'checkpoint_path',
                        action = 'store',type = str,default = 'summaries',
                        help = 'Path to checkpoint.')
    parser.add_argument('--batch_size',dest = 'batch_size',
                        action = 'store',type = int,default = 4,
                        help = 'Size of mini batch.')
    args = parser.parse_args()

    quality_net = keras.models.load_model(args.checkpoint_path)

    def generator():
        G = image_generator_slide(
            args.slide_path,args.input_height,args.input_width)
        for image,coords in G:
            image = image / 255.
            yield image,coords

    output_types = (tf.float32,tf.string)
    output_shapes = (
        [args.input_height,args.input_width,n_channels],[])
    tf_dataset = tf.data.Dataset.from_generator(
        generator,output_types=output_types,output_shapes=output_shapes)
    tf_dataset = tf_dataset.batch(args.batch_size,drop_remainder=False)
    tf_dataset = tf_dataset.prefetch(5)

    for image,coords in tqdm(tf_dataset):
        prediction = quality_net(image)
        for c,p in zip(coords.numpy(),prediction.numpy()):
            print('OUT,{},{},{}'.format(c.decode(),int(p>0.5),float(p)))