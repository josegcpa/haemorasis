import os
"""
Functions for QualityNet
More info in https://github.com/josegcpa/quality-net
"""

import numpy as np
import h5py
from PIL import Image
from glob import glob

import tensorflow as tf
from tensorflow import keras

def quality_net_model(defined_model,h,w):
    prediction_layer = keras.Sequential(
        [keras.layers.Dense(256,activation='relu'),
         keras.layers.Dense(1,'sigmoid')])

    inputs = tf.keras.Input(shape=(h, w, 3))
    x = defined_model(inputs)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model

class ColourAugmentation(keras.layers.Layer):
    def __init__(self,
                 brightness_delta,
                 contrast_lower,contrast_upper,
                 hue_delta,
                 saturation_lower,saturation_upper,
                 min_jpeg_quality,max_jpeg_quality,
                 probability=0.1):
        super(ColourAugmentation,self).__init__()
        self.probability = probability
        self.brightness_delta = brightness_delta
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper
        self.hue_delta = hue_delta
        self.saturation_lower = saturation_lower
        self.saturation_upper = saturation_upper
        self.min_jpeg_quality = min_jpeg_quality
        self.max_jpeg_quality = max_jpeg_quality
        
    def brightness(self,x):
        return tf.image.random_brightness(
            x,self.brightness_delta)
    
    def contrast(self,x):
        return tf.image.random_contrast(
            x,self.contrast_lower,self.contrast_upper)
    
    def hue(self,x):
        return tf.image.random_hue(
            x,self.hue_delta)
    
    def saturation(self,x):
        return tf.image.random_saturation(
            x,self.saturation_lower,self.saturation_upper)

    def jpeg_quality(self,x):
        if (self.max_jpeg_quality - self.min_jpeg_quality) > 0:
            return tf.image.random_jpeg_quality(
                x,self.min_jpeg_quality,self.max_jpeg_quality)
        else:
            return x

    def call(self,x):
        fn_list = [self.brightness,self.contrast,
                   self.hue,self.saturation]
        np.random.shuffle(fn_list)
        for fn in fn_list:
            if np.random.uniform() < self.probability:
                x = fn(x)
        if np.random.uniform() < self.probability:
            x = jpeg_quality(x)
        x = tf.clip_by_value(x,0,1)
        return x
    
class Flipper(keras.layers.Layer):
    def __init__(self,probability=0.1):
        super(Flipper,self).__init__()
        self.probability = probability
            
    def call(self,x):
        if np.random.uniform() < self.probability:
            x = tf.image.flip_left_right(x)
        if np.random.uniform() < self.probability:
            x = tf.image.flip_up_down(x)
        return x

class ImageCallBack(keras.callbacks.Callback):
    def __init__(self,save_every_n,tf_dataset,log_dir):
        super(ImageCallBack, self).__init__()
        self.save_every_n = save_every_n
        self.tf_dataset = iter(tf_dataset)
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.count = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.count % self.save_every_n == 0:
            batch = next(self.tf_dataset)
            y_augmented,y_true = batch
            prediction = self.model.predict(y_augmented)
            with self.writer.as_default():
                tf.summary.image("0:InputImage",y_augmented,self.count)
                tf.summary.image("1:GroundTruth",y_true,self.count)
                tf.summary.image("2:Prediction",prediction,self.count)
                tf.summary.scalar("Loss",logs['loss'],self.count)
                tf.summary.scalar("MAE",logs['mean_absolute_error'],self.count)
        self.count += 1

class DataGenerator:
    def __init__(self,hdf5_path,shuffle=True,transform=None):
        self.hdf5_path = hdf5_path
        self.h5 = h5py.File(self.hdf5_path,'r')
        self.shuffle = shuffle
        self.transform = transform
        self.all_keys = list(self.h5.keys())
        self.n_images = len(self.all_keys)
    
    def generate(self,with_path=False):
        image_idx = [x for x in range(self.n_images)]
        if self.shuffle == True:
            np.random.shuffle(image_idx)
        for idx in image_idx:
            P = self.all_keys[idx]
            x = self.h5[P]['image'][:,:,:3]
            c = [float(self.h5[P]['class'][()])]
            x = tf.convert_to_tensor(x) / 255
            if self.transform is not None:
                x = self.transform(x)
            yield x,c

class LargeImage:
    def __init__(self,image,tile_size=[512,512],
                 output_channels=3,offset=0):
        """
        Class facilitating the prediction for large images by 
        performing all the necessary operations - tiling and 
        reconstructing the output.
        """
        self.image = image
        self.tile_size = tile_size
        self.output_channels = output_channels
        self.offset = offset
        self.h = self.tile_size[0]
        self.w = self.tile_size[1]
        self.sh = self.image.shape[:2]
        self.output = np.zeros([self.sh[0],self.sh[1],self.output_channels])
        self.denominator = np.zeros([self.sh[0],self.sh[1],1])

    def tile_image(self):
        for x in range(0,self.sh[0]+self.offset,self.h):
            x = x - self.offset
            if x + self.tile_size[0] > self.sh[0]:
                x = self.sh[0] - self.tile_size[0]
            for y in range(0,self.sh[1]+self.offset,self.w):
                y = y - self.offset
                if y + self.tile_size[1] > self.sh[1]:
                    y = self.sh[1] - self.tile_size[1]
                x_1,x_2 = x, x+self.h
                y_1,y_2 = y, y+self.w
                yield self.image[x_1:x_2,y_1:y_2,:],((x_1,x_2),(y_1,y_2))

    def update_output(self,image,coords):
        (x_1,x_2),(y_1,y_2) = coords
        self.output[x_1:x_2,y_1:y_2,:] += image
        self.denominator[x_1:x_2,y_1:y_2,:] += 1

    def return_output(self):
        return self.output/self.denominator

class Accuracy(keras.metrics.Accuracy):
    # adapts Accuracy to work with model.fit using logits
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.where(
            y_pred > 0.5,tf.ones_like(y_pred),tf.zeros_like(y_pred))
        return super().update_state(y_true,y_pred,sample_weight)

class Precision(tf.keras.metrics.Precision):
    # adapts Precision to work with model.fit using logits
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.where(
            y_pred > 0.5,tf.ones_like(y_pred),tf.zeros_like(y_pred))
        return super().update_state(y_true,y_pred,sample_weight)

class Recall(tf.keras.metrics.Recall):
    # adapts Sensitivity to work with model.fit using logits
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.where(
            y_pred > 0.5,tf.ones_like(y_pred),tf.zeros_like(y_pred))
        return super().update_state(y_true,y_pred,sample_weight)