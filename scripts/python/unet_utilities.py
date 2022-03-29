"""
Deep learning/TF-related operations for U-Net.
More details in https://github.com/josegcpa/u-net-tf2
"""

import os
import numpy as np
from math import inf
import cv2
import tifffile as tiff
import h5py
from scipy.spatial import distance
from PIL import Image

import tensorflow as tf
from tensorflow import keras


def safe_log(tensor):
    """
    Prevents log(0)

    Arguments:
    * tensor - tensor
    """
    return tf.log(tf.clip_by_value(tensor,1e-32,tf.reduce_max(tensor)))

class UNetConvLayer(keras.layers.Layer):
    def __init__(self,
                 depth,
                 conv_size,
                 stride,
                 factorization=False,
                 padding='VALID',
                 dropout_rate=0.2,
                 beta=0.005):
        
        super(UNetConvLayer,self).__init__()
        self.depth = depth
        self.conv_size = conv_size
        self.stride = stride
        self.factorization = factorization
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.setup_layers()

    def setup_layers(self):
        self.layer = keras.Sequential()
        if self.factorization == False:
            self.layer.add(keras.layers.Conv2D(
                self.depth,self.conv_size,strides=self.stride,
                padding=self.padding,
                kernel_regularizer=keras.regularizers.l2(self.beta)))
        
        if self.factorization == True:
            self.layer.add(keras.layers.Conv2D(
                self.depth,[1,self.conv_size],
                strides=self.stride,padding=self.padding,
                kernel_regularizer=keras.regularizers.l2(self.beta)))
            self.layer.add(keras.layers.Conv2D(
                self.depth,[self.conv_size,1],
                strides=self.stride,padding=self.padding,
                kernel_regularizer=keras.regularizers.l2(self.beta)))
        self.layer.add(keras.layers.LeakyReLU())
        self.layer.add(keras.layers.Dropout(self.dropout_rate))
    
    def call(self,x):
        return self.layer(x)

class UNetConvBlock(keras.layers.Layer):
    def __init__(self,
                 depth,
                 conv_size,
                 stride=1,
                 factorization=False,
                 padding='VALID',
                 dropout_rate=0.2,
                 beta=0.005):
        
        super(UNetConvBlock,self).__init__()
        self.depth = depth
        self.conv_size = conv_size
        self.stride = stride
        self.factorization = factorization
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.setup_layers()

    def setup_layers(self):
        self.layer = keras.Sequential()
        self.layer.add(UNetConvLayer(
            self.depth,self.conv_size,self.stride,self.factorization,
            self.padding,self.dropout_rate,self.beta))
        self.layer.add(UNetConvLayer(
            self.depth,self.conv_size,1,self.factorization,
            self.padding,self.dropout_rate,self.beta))
    
    def call(self,x):
        return self.layer(x)

class UNetReductionBlock(keras.layers.Layer):
    def __init__(self,
                 depth,
                 conv_size,
                 stride=1,
                 factorization=False,
                 padding='VALID',
                 dropout_rate=0.2,
                 beta=0.005):
        
        super(UNetReductionBlock,self).__init__()
        self.depth = depth
        self.conv_size = conv_size
        self.stride = stride
        self.factorization = factorization
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.setup_layers()

    def setup_layers(self):
        self.layer_pre = UNetConvBlock(
            self.depth,self.conv_size,self.stride,self.factorization,
            self.padding,self.dropout_rate)
        self.layer_down = keras.layers.MaxPool2D([2,2],2)
    
    def call(self,x):
        pre = self.layer_pre(x)
        down = self.layer_down(pre)
        return down,pre

class UNetReconstructionBlock(keras.layers.Layer):
    def __init__(self,
                 depth,
                 conv_size,
                 stride=1,
                 factorization=False,
                 padding='VALID',
                 dropout_rate=0.2,
                 beta=0.005):
        
        super(UNetReconstructionBlock,self).__init__()
        self.depth = depth
        self.conv_size = conv_size
        self.stride = stride
        self.factorization = factorization
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.setup_layers()

    def setup_layers(self):
        self.layer_pre = keras.Sequential()
        self.layer_pre.add(keras.layers.Conv2DTranspose(
            self.depth,3,strides=2,padding='same',
            kernel_regularizer=keras.regularizers.l2(self.beta)))
        self.concat_op = keras.layers.Concatenate(axis=-1)
        self.layer_post = UNetConvBlock(
            self.depth,self.conv_size,self.stride,self.factorization,
            self.padding,self.dropout_rate,self.beta)
    
    def crop_x_to_y(self,x,y):
        fraction = x.shape[2] / y.shape[2]
        if fraction == 1.:
            return x
        else:
            return tf.image.central_crop(x,fraction)

    def call(self,x,y=None):
        x = self.layer_pre(x)
        if y is not None:
            y = self.crop_x_to_y(y,x)
        if y is not None:
            x = self.concat_op([x,y])
        return self.layer_post(x)

class ChannelSqueezeAndExcite(keras.layers.Layer):
    def __init__(self,n_channels,beta):
        self.n_channels = n_channels
        self.setup_layers()
        self.beta = beta

    def setup_layers(self):
        self.layer = keras.Sequential([
            keras.layers.Dense(self.n_channels),
            keras.layers.Activation('relu'),
            keras.layers.Dense(self.n_channels),
            keras.layers.Activation('sigmoid')])
    
    def call(self,x):
        squeezed_input = tf.math.reduce_mean(x,[1,2])
        excited_input = self.layer(squeezed_input)
        excited_input = tf.expand_dims(excited_input,1)
        excited_input = tf.expand_dims(excited_input,1)
        return excited_input * x

class SpatialSqueezeAndExcite(keras.layers.Layer):
    def __init__(self):
        self.setup_layers()
    
    def setup_layers(self):
        self.layer = keras.Sequential([
            keras.layers.Conv2D(1,1,padding='same'),
            keras.layers.Activation('sigmoid')])
    
    def call(self,x):
        return self.layer(x) * x

class SCSqueezeAndExcite(keras.layers.Layer):
    def __init__(self,n_channels):
        self.n_channels = n_channels
        self.setup_layers()

    def setup_layers(self,x):
        self.spatial_sae = SpatialSqueezeAndExcite()
        self.channel_sae = ChannelSqueezeAndExcite(self.n_channels)

    def call(self,x):
        return self.spatial_sae(x) + self.channel_sae(x)

class UNet(keras.Model):
    def __init__(self,
                 depth_mult=1.,
                 padding='VALID',
                 factorization=False,
                 n_classes=2,
                 beta=0.005,
                 squeeze_and_excite=False,
                 dropout_rate=0.2,
                 loss_fn=None):
        super(UNet, self).__init__()
        self.depth_mult = depth_mult
        self.padding = padding
        self.factorization = factorization
        self.n_classes = n_classes
        self.beta = beta
        self.squeeze_and_excite = squeeze_and_excite
        self.dropout_rate = dropout_rate
        self.loss_fn = loss_fn # used in train_step

        self.depths = [64,128,256,512,1024]
        self.depths = [int(x*self.depth_mult) for x in self.depths]
        self.setup_network()

    def setup_network(self):
        self.reductions = [
            UNetConvBlock(
                depth=self.depths[0],conv_size=3,
                factorization=self.factorization,padding=self.padding,
                dropout_rate=self.dropout_rate,beta=self.beta)]
        for depth in self.depths[1:-1]:
            m = UNetReductionBlock(
                depth=depth,conv_size=3,
                factorization=self.factorization,padding=self.padding,
                dropout_rate=self.dropout_rate,beta=self.beta)
            self.reductions.append(m)

        self.bottleneck_layer_1 = UNetConvBlock(
            depth=self.depths[-2],conv_size=3,stride=1,
            factorization=self.factorization,padding=self.padding,
            dropout_rate=self.dropout_rate,beta=self.beta)
        self.bottleneck_layer_2 = UNetConvLayer(
            depth=self.depths[-1],conv_size=3,stride=2,
            factorization=self.factorization,padding=self.padding,
            dropout_rate=self.dropout_rate,beta=self.beta)

        self.reconstructions = []
        for depth in self.depths[-2::-1]:
            m = UNetReconstructionBlock(
                depth=self.depths[0],conv_size=3,
                factorization=self.factorization,padding=self.padding,
                dropout_rate=self.dropout_rate,beta=self.beta)
            self.reconstructions.append(m)
        
        self.classification_layer = keras.layers.Conv2D(
            self.n_classes,1)
    
    def call(self,x):
        red_1 = self.reductions[0](x)
        red_2,pre_1 = self.reductions[1](red_1)
        red_3,pre_2 = self.reductions[2](red_2)
        red_4,pre_3 = self.reductions[3](red_3)

        pre_4 = self.bottleneck_layer_1(red_4)
        red_5 = self.bottleneck_layer_2(pre_4)
        
        rec_1 = self.reconstructions[0](red_5,pre_4)
        rec_2 = self.reconstructions[1](rec_1,pre_3)
        rec_3 = self.reconstructions[2](rec_2,pre_2)
        rec_4 = self.reconstructions[3](rec_3,pre_1)

        classification = self.classification_layer(rec_4)

        return classification

    def train_step(self, data):
        x, y, w = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_fn(
                y,y_pred,w,regularization_losses=self.losses)

        # ensures loss in metrics is also updated
        self.compiled_loss(
            y,y_pred,sample_weight=None,regularization_losses=self.losses) 

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

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
        for x in range(0,self.sh[0],self.h-self.offset):
            if x + self.tile_size[0] > self.sh[0]:
                x = self.sh[0] - self.tile_size[0]
            for y in range(0,self.sh[1],self.w-self.offset):
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