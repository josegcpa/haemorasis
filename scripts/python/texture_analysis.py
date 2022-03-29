"""
Set of functions to characterise texture.
"""

import numpy as np
from skimage.feature import greycoprops
from skimage.feature import greycomatrix

def quantize_image(image):
    q_image = np.uint8(np.mean(image,axis=2) * 32)
    q_image = np.clip(q_image,a_min=0,a_max=31)
    return q_image

def glcm(image,x,y):
    gray_matrix = np.zeros([32,32])
    avail_coord_list = [(x1,y1) for x1,y1 in zip(x,y)]
    for x1,y1 in zip(x,y):
        coord_list = [(x1+i,y1) for i in range(-4,5) if i != 0]
        coord_list.extend([(x1,y1+i) for i in range(-4,5) if i != 0])
        coord_list = [c for c in coord_list if c in avail_coord_list]
        for c in coord_list:
            a,b = image[x1,y1],image[c[0],c[1]]
            gray_matrix[a,b] += 1
    gray_matrix = gray_matrix / np.sum(gray_matrix)
    return gray_matrix

def texture_features(image,x,y):
    mask = np.ones_like(image,dtype=np.bool)
    mask[x,y] = False
    image[mask] = 32
    gray_matrix = greycomatrix(
        image,distances=[1,2,3,4],angles=[0,np.pi/2,np.pi,-np.pi/2],
        symmetric=True,levels=33
    )
    gray_matrix = gray_matrix[:32,:,:,:]
    gray_matrix = gray_matrix[:,:32,:,:]
    contrast = greycoprops(gray_matrix,'contrast').mean()
    energy = greycoprops(gray_matrix,'energy').mean()
    homogeneity = greycoprops(gray_matrix,'homogeneity').mean()
    correlation = greycoprops(gray_matrix,'correlation').mean()
    del image
    return [np.squeeze(contrast),np.squeeze(energy),
            np.squeeze(homogeneity),np.squeeze(correlation)]

def wrapper(image,x,y):
    out = {}
    text_feat = texture_features(image,x,y)
    out['contrast'] = text_feat[0]
    out['energy'] = text_feat[1]
    out['homogeneity'] = text_feat[2]
    out['correlation'] = text_feat[3]

    return out
