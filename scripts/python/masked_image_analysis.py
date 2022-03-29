"""
Functions for masked image analysis 
(uses an image and its mask, retrieves connected objects
and characterises each object)
"""

import numpy as np
import cv2
import time
from multiprocessing import Pool

import shape_analysis as sa
import color_distribution_analysis as cda
import texture_analysis as ta

def calculate_all_features(image_qimage_x_y_cnt_mask):
    image,q_image,x,y,cnt,mask = image_qimage_x_y_cnt_mask
    shape_features = sa.wrapper(x,y,cnt)
    color_features = cda.wrapper(image,cnt,x,y)
    texture_features = ta.wrapper(q_image,x,y)
    image_snippet = (image*np.expand_dims(mask,2))[x.min():x.max(),y.min():y.max(),:]
    image_snippet_whole = image[x.min():x.max(),y.min():y.max(),:]
    return {'isolated_image':image_snippet,
            'image':image_snippet_whole,
            'x':x,
            'y':y,
            **shape_features,
            **color_features,
            **texture_features}

def wrapper_single_image_separate(image,mask,cnt):
    x,y = np.where(mask > 0)
    shape_features = sa.wrapper_separate(x,y,cnt)
    color_features = cda.wrapper_separate(
        image,np.concatenate(cnt,axis=0),x,y)
    return {'x':x,
            'y':y,
            **shape_features,
            **color_features}

def wrapper_single_image(image,mask,cnt):
    q_image = ta.quantize_image(image)
    x,y = np.where(mask > 0)
    #a = time.time()
    shape_features = sa.wrapper(x,y,cnt)
    #b = time.time()
    color_features = cda.wrapper(image,cnt,x,y)
    #c = time.time()
    texture_features = ta.wrapper(q_image,x,y)
    #d = time.time()
    return {'x':x,
            'y':y,
            **shape_features,
            **color_features,
            **texture_features}

def wrapper(image,mask,preprocessing=False,nproc=1):
    kernel = np.ones([3,3])
    image = image / 255.
    q_image = ta.quantize_image(image)
    ret, labels = cv2.connectedComponents(mask[:,:,0])
    h,w = labels.shape
    if nproc > 1:
        pool = Pool(nproc)
        all_xy_cnt = []
        for i in range(1,ret):
            mask = np.zeros([h,w],dtype=np.uint8)
            mask[labels == i] = 255
            if preprocessing == True:
                mask = cv2.morphologyEx(
                    mask,cv2.MORPH_OPEN,kernel,iterations=2)
            x,y = np.where(mask == 255)
            cnt = cv2.findContours(mask,mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_NONE)[0][0]
            if len(x) > 20:
                if np.sum((x==0) + (x==(h-1)) + (y==0) + (y==(w-1))) < 5:
                    all_xy_cnt.append([image,q_image,x,y,cnt,mask])

        all_features = pool.map(calculate_all_features,all_xy_cnt)

    else:
        all_features = []
        for i in range(1,ret):
            mask = np.zeros([h,w],dtype=np.uint8)
            mask[labels == i] = 255
            x,y = np.where(mask == 255)
            cnt = cv2.findContours(mask,mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_NONE)[0][0]
            if len(x) > 5:
                if np.sum((x==0) + (x==h) + (y==0) + (y==w)) < 5:
                    full_dict = calculate_all_features([image,q_image,x,y,cnt,mask])
                    all_features.append(full_dict)

    return all_features
