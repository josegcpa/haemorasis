"""
Script to segment WBC and RBC in a whole blood slide from a quality CSV.

Usage:
    python3 segment_slide_wbc_rbc.py --help
"""

import numpy as np
import argparse
import h5py
import cv2
from scipy.ndimage import convolve
from skimage.filters import apply_hysteresis_threshold
from scipy.ndimage.morphology import binary_fill_holes
from multiprocessing import Queue,Process
from tqdm import tqdm

from tensorflow.python.framework.errors_impl import InvalidArgumentError

import tensorflow as tf

# prevents greedy memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

from image_generator import ImageGeneratorWithQueue
from unet_utilities import *
import MIA
MIA_FEATURES = MIA.MIA_FEATURES

# process-specific functions

class BCProcess:
    def __init__(self,F_path,h,w,extra):
        self.F_path = F_path
        self.h = h
        self.w = w
        self.extra = extra
        self.N_cells = 0
        self.N_images = 0
        self.Centers = {}

    def generate(self,q):
        C = True
        while C == True:
            element = q.get()
            if element is not None:
                self.process_element(element)
            else:
                C = False
        self.F.attrs['NCells'] = self.N_cells
        self.F.attrs['NImages'] = self.N_images
        self.close_hdf5()

    def init_hdf5(self):
        self.F = h5py.File(self.F_path,'w')
    
    def close_hdf5(self):
        self.F.close()

class WBCProcess(BCProcess):
    def process_element(self,element):
        self.N_images += 1
        image,mask,rescale_factor,coords = element
        h,w,_ = image.shape
        g = refine_prediction_wbc(image,mask,rescale_factor)
        for obj in g:
            y,x = obj['x'],obj['y']
            if np.any([x.min()==0,x.max()==h,y.min()==0,y.max()==w]):
                pass
            else:
                x += coords[0]
                y += coords[1]
                C = str([np.mean(x),np.mean(y)])
                C_dict = str(
                    [np.round(np.mean(x)/8),np.round(np.mean(y)/8)])
                if C_dict not in self.Centers:
                    self.Centers[C_dict] = 1
                    g = self.F.create_group(str(C))
                    image = obj['image']
                    mask = obj['mask']
                    cnt = obj['cnt']

                    g.create_dataset(
                        "X",x.shape,dtype=np.int32,data=x)
                    g.create_dataset(
                        "Y",y.shape,dtype=np.int32,data=y)
                    g.create_dataset(
                        "image",image.shape,dtype=np.uint8,data=image)
                    g.create_dataset(
                        "mask",mask.shape,dtype=np.uint8,data=mask)
                    g.create_dataset(
                        "cnt",cnt.shape,dtype=np.int32,data=cnt)

                    self.N_cells += 1
                    self.F.flush()

class RBCProcess(BCProcess):
    def process_element(self,element):
        self.N_images += 1
        image,rescale_factor,coords = element
        h,w,_ = image.shape
        _,mask = mask_rbc(image,rescale_factor)
        features = refine_prediction_rbc(image,mask)
        for obj in features:
            y,x = obj['x'],obj['y']
            if np.any([x.min()==0,x.max()==h,y.min()==0,y.max()==w]):
                pass
            else:
                x += coords[0]
                y += coords[1]
                C = str([np.mean(x),np.mean(y)])
                C_dict = str(
                    [np.round(np.mean(x)/8),np.round(np.mean(y)/8)])
                if C_dict not in self.Centers:
                    self.Centers[C_dict] = 1
                    g = self.F.create_group(str(C))
                    image = obj['image']
                    mask = obj['mask']
                    cnt = obj['cnt']

                    g.create_dataset(
                        "X",x.shape,dtype=np.int32,data=x)
                    g.create_dataset(
                        "Y",y.shape,dtype=np.int32,data=y)
                    g.create_dataset(
                        "image",image.shape,dtype=np.uint8,data=image)
                    g.create_dataset(
                        "mask",mask.shape,dtype=np.uint8,data=mask)
                    g.create_dataset(
                        "cnt",cnt.shape,dtype=np.int32,data=cnt)    
                    self.N_cells += 1
                    self.F.flush()

class ProcessingQueue:
    def __init__(self,fn):
        self.fn = fn
        self.q = Queue(maxsize=500)
        self.p = Process(target=self.fn,args=((self.q),))

    def add_to_queue(self,x):
        self.q.put(x)
    
    def start(self):
        self.p.daemon = True
        self.p.start()

    def join(self):
        self.p.join()

# WBC segmentation post-processing

F_size = 9
Filter = np.ones([F_size,F_size]) / (F_size**2)

def mask_wbc(x,model,tta=False):
    x = tf.expand_dims(x,axis=0)
    if tta == True:
        rotated_x = [x,tf.image.rot90(x,1),tf.image.rot90(x,2),
                     tf.image.rot90(x,3)]
        x = tf.concat(rotated_x,axis=0)
        pred = model.predict(x)
        rot_pred = [
            pred[0,:,:,:],
            tf.image.rot90(pred[1,:,:,:],-1),
            tf.image.rot90(pred[2,:,:,:],-2),
            tf.image.rot90(pred[3,:,:,:],-3)]
        rot_pred = tf.stack(rot_pred,axis=0)
        rot_pred = tf.reduce_mean(rot_pred,axis=0)
        prediction = tf.nn.softmax(rot_pred,-1)[:,:,1]
    else:
        prediction = model.predict(x)
        prediction = tf.nn.softmax(prediction,-1)[0,:,1]
    return prediction.numpy()

def convolve_n(image,F,n=1,thr=0.6):
    for i in range(n):
        image = convolve(image.astype(np.float32),F) > thr
    return image.astype(np.bool)

def draw_hulls(image):
    contours,_ = cv2.findContours(image.astype(np.uint8),2,1)
    cnt = contours[-1]
    defects = cv2.convexityDefects(
        cnt, cv2.convexHull(cnt,returnPoints = False))
    if defects is not None:
        defects = defects[defects[:,0,-1]>2000,:,:]
        if defects.size > 0:
            for defect in defects:
                a = tuple([x for x in cnt[defect[0,0],0,:]])
                b = tuple([x for x in cnt[defect[0,1],0,:]])
                output = cv2.line(image.astype(np.uint8),a,b,1)
                output = binary_fill_holes(output)
        else:
            output = image
    else:
        output = image
    return output,cnt

def filter_refine_cell(image_labels_im_i):
    image,labels_im,i,RS = image_labels_im_i
    s = 128
    s_2 = s // 2
    sh = image.shape
    x,y = np.where(labels_im == i)
    R = x.min(),y.min(),x.max(),y.max()
    cx,cy = int((R[2]+R[0])/2),int((R[3]+R[1])/2)
    sx,sy = R[2]-R[0],R[3]-R[1]
    cc = [cx-s_2,cy-s_2]
    cc.extend([cc[0]+s,cc[1]+s])
    features = None
    conditions = [
        cc[0]<0,cc[1]<0,cc[2]>sh[0],cc[3]>sh[1],
        sx > 120,sy > 120]
    if np.any(conditions):
        pass
    else:
        try:
            x_ = x - cc[0]
            y_ = y - cc[1]
            S = len(x)
            if (S > 1000/RS) and (S < 8000/RS):
                sub_image = image[cc[0]:cc[2],cc[1]:cc[3],:]
                mask_binary_holes = np.zeros([s,s])
                mask_binary_holes[(x_,y_)] = 1
                mask_convolved = convolve_n(
                    mask_binary_holes.astype(np.float32),
                    Filter,3,thr=0.5)
                mask_hulls,cnt = draw_hulls(mask_convolved)
                mask_hulls = binary_fill_holes(mask_hulls)
                features = {}
                features['image'] = sub_image
                features['mask'] = mask_hulls
                features['cnt'] = cnt
                features['x'] = x
                features['y'] = y
        except:
            features = None
    return features

def refine_prediction_wbc(image,mask,rescale_factor):
    mask_binary = apply_hysteresis_threshold(mask,0.5,0.5)
    mask_binary = np.where(mask > 0.5,1,0)
    mask_binary_holes = binary_fill_holes(mask_binary)

    num_labels, labels_im = cv2.connectedComponents(
        np.uint8(mask_binary_holes))

    if num_labels > 1:
        for i in range(1,num_labels):
            features = filter_refine_cell([image,labels_im,i,rescale_factor])
            if features is not None:
                yield features
            pass

def refine_prediction_rbc(image,mask):
    ret, labels_im = cv2.connectedComponents(mask[:,:,0])
    h,w = labels_im.shape
    m = 16
    all_features = []
    for i in range(1,ret):
        x,y = np.where(labels_im == i)
        if np.sum((x==0) + (x==h) + (y==0) + (y==w)) < 5:
            R = x.min(),y.min(),x.max(),y.max()
            sx,sy = R[2]-R[0]+m*2,R[3]-R[1]+m*2
            cc = [R[0]-m,R[1]-m]
            cc.extend([cc[0]+sx,cc[1]+sy])
            x_ = x - cc[0]
            y_ = y - cc[1]
            S = len(x)
            if S > 100:
                sub_image = image[cc[0]:cc[2],cc[1]:cc[3],:]
                if sub_image.shape[0] > 10 and sub_image.shape[1] > 10:
                    mask_binary_holes = np.zeros([sx,sy])
                    mask_binary_holes[(x_,y_)] = 1
                    cnt,_ = cv2.findContours(
                        mask_binary_holes.astype(np.uint8),2,1)
                    cnt = cnt[-1]
                    features = {}
                    features['image'] = sub_image
                    features['mask'] = mask_binary_holes
                    features['cnt'] = cnt
                    features['x'] = x
                    features['y'] = y
                    all_features.append(features)
    return all_features

# RBC segmentation and post-processing

from mask_rbc import wraper as mask_rbc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'segment_slide_wbc_rbc.py',
        description = 'Segments RBC and WBC in a slide given a quality CSV.')

    parser.add_argument('--csv_path',dest='csv_path',
                        action='store',default=None,
                        help='Path to CSV file with the quality file.')
    parser.add_argument('--slide_path',dest='slide_path',
                        action='store',default=None,
                        help="Path to whole blood slide")
    parser.add_argument('--unet_checkpoint_path',dest='unet_checkpoint_path',
                        action='store',default=None,
                        help='Path to U-Net checkpoint.')
    parser.add_argument('--depth_mult',dest='depth_mult',
                        action='store',type=float,default=1.0,
                        help='Depth of the U-Net.')
    parser.add_argument('--wbc_output_path',dest='wbc_output_path',
                        action='store',default=None,
                        help='Output path for the WBC hdf5 file.')
    parser.add_argument('--rbc_output_path',dest='rbc_output_path',
                        action='store',default=None,
                        help='Output path for the RBC hdf5 file.')
    parser.add_argument('--rescale_factor',dest='rescale_factor',
                        action='store',type=float,default=1,
                        help='Factor to resize cells (due to different mpp).')

    args = parser.parse_args()

    h,w,extra = 512,512,128
    new_h,new_w = h+extra,w+extra

    igwq = ImageGeneratorWithQueue(args.slide_path,args.csv_path,
                                extra_padding=extra//2,
                                maxsize=25)
    igwq.start()

    # instantiate U-Net and load it
    u_net = UNet(depth_mult=args.depth_mult,padding='SAME',
                factorization=False,n_classes=2,
                dropout_rate=0,squeeze_and_excite=False)
    u_net.load_weights(args.unet_checkpoint_path)
    u_net.trainable = False
    u_net.make_predict_function()

    wbc_process = WBCProcess(args.wbc_output_path,h,w,extra)
    wbc_process.init_hdf5()
    rbc_process = RBCProcess(args.rbc_output_path,h,w,extra)
    rbc_process.init_hdf5()

    def generator():
        G = igwq.generate()
        for image,coords in G:
            image = image / 255.
            yield np.float32(image),np.array(coords,dtype=np.int32)

    output_types = (tf.float32,tf.int32)
    output_shapes = (
        [new_h,new_w,3],[2])
    tf_dataset = tf.data.Dataset.from_generator(
        generator,output_types=output_types,output_shapes=output_shapes)
    tf_dataset = tf_dataset.batch(2,drop_remainder=True)
    tf_dataset = tf_dataset.prefetch(36)

    for image,coords in tqdm(tf_dataset):
        coords = list(coords.numpy())
        normalised_input = image
        for n_i,c in zip(normalised_input,coords):
            masked_wbc = mask_wbc(n_i,u_net,tta=True)
            n_i_p = n_i.numpy()
            n_i_p = np.uint8(n_i_p*255)

            wbc_process.process_element([n_i_p,masked_wbc,args.rescale_factor,c])
            rbc_process.process_element([n_i_p,args.rescale_factor,c])

    wbc_process.close_hdf5()
    rbc_process.close_hdf5()
