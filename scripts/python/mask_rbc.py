"""
Functions for RBC masking
"""

from math import pi
from glob import glob
from itertools import permutations
import pickle
import os
import sys
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from skimage.io import imread
from skimage.feature import canny,peak_local_max,blob_doh
from skimage.segmentation import watershed
from skimage import filters
import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(te - ts,'seconds')
        return result
    return timed

def image_to_array(image_path):
    return np.array(Image.open(image_path))

def rgb_to_hsv(rgb_image):
    return cv2.cvtColor(rgb_image,cv2.COLOR_RGB2HLS_FULL)

def rgb_to_hsv(rgb_image):
    return cv2.cvtColor(rgb_image,cv2.COLOR_RGB2HSV)

def rgb_to_lab(rgb_image):
    return cv2.cvtColor(rgb_image,cv2.COLOR_RGB2LAB)

def thresholding(img):
    return cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

def mask_wbc(im_array):
     return (im_array[:,:,0]/(im_array[:,:,2] + 1) > 1).astype(np.uint8)

def mask_bg(im_array):
    return (im_array[:,:,0]/(im_array[:,:,1] + 1) > 1.1).astype(np.uint8)

def fill_holes(img):
    out = img.copy()
    input_image = img.copy()

    im2,contours,hierarchy = cv2.findContours(
        input_image.astype(np.uint8),
        mode = cv2.RETR_LIST,
        method = cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(out, contours, -1, (255,255,255), -1)
    out[0:10,0:10] = 0
    h, w = out.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(out, mask, (1,1), 255)

    mask = cv2.morphologyEx(mask[1:-1,1:-1] * 255,cv2.MORPH_OPEN,np.ones([9,9]))
    mask = np.stack((mask,mask,mask),-1)

    return mask

def gamma_stretching(image,gamma=2):
    if 'int' in str(image.dtype):
        image = image/255.
    image = image ** gamma
    return image

def linear_stretching(image):
    image = image - image.min() / (image.max() - image.min())
    image *= 255
    image = image.astype('uint8')
    return image

def get_edges(img):
    #mask = mask_bg(img)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) * mask
    otsu_img = cv2.dilate(otsu_threshold(img),kernel=np.ones((3,3)))
    img = np.uint8(gamma_stretching(img,5) * 255)
    img = linear_stretching(img)
    img = cv2.medianBlur(img,5)
    img = cv2.GaussianBlur(img,(3,3),1)
    img = -cv2.bilateralFilter(img,11,5,10)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img,50,150)
    edges = cv2.dilate(edges,kernel=np.ones((5,5)),iterations=1)
    edges = cv2.erode(edges,kernel=np.ones((3,3)),iterations=1)
    edges = np.where(edges > 0,1,0)
    edges_otsu = edges * otsu_img
    if edges_otsu.max() - edges_otsu.min() != 0:
        edges = linear_stretching(edges_otsu)
    return edges

def get_contours(img):
    cnt,nc = cv2.findContours(
        np.uint8(img).copy(),
        mode = cv2.RETR_EXTERNAL,
        method = cv2.CHAIN_APPROX_NONE)
    return cnt

def filter_contours(contours,min_poly,min_area,max_area):
    contour_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > min_area) & (area < max_area):
            polygon = cv2.approxPolyDP(
                contour,0.01 * cv2.arcLength(contour,True),True
            )
            if (len(polygon) > min_poly):
                contour_list.append(contour)
    return contour_list

def filter_contours_color(contours,image):
    def masked_mean(image,mask):
        image = image.copy() * mask
        r,b,g = image[:,:,0],image[:,:,1],image[:,:,2]
        avgs = [np.mean(r[r.nonzero()]),
                np.mean(b[b.nonzero()]),
                np.mean(g[g.nonzero()])]
        return avgs
    filtered_contours = []
    mask = np.zeros(image.shape)
    for i in range(len(contours)):
        cv2.drawContours(mask,contours,i,color = (1,1,1),thickness = -1)
        avgs = masked_mean(image,mask)
        cv2.drawContours(mask,contours,i,color = (0,0,0),thickness = -1)
        if avgs[0] / avgs[1] > 1.2:
            filtered_contours.append(contours[i])
    return filtered_contours

def filter_contours_color_2(contours,image):
    def masked_mean(image,mask):
        return [cv2.mean(image[:,:,0],mask = mask),
                cv2.mean(image[:,:,1],mask = mask),
                cv2.mean(image[:,:,2],mask = mask)]
    filtered_contours = []
    for i in range(len(contours)):
        mask = np.zeros(image[:,:,0].shape)
        cv2.drawContours(mask,contours,i,color = (1,1,1),thickness = -1)
        avgs = masked_mean(image,mask)
        print(avgs)
        if avgs[0] / avgs[1] > 1.2:
            filtered_contours.append(contours[i])
    return filtered_contours

def filter_contours_size_color(contours,min_poly,min_area,max_area,image):
    """
    A function to perform both filterings in a 'single' loop.
    """
    def masked_mean(image,mask):
        image = image.copy() * mask
        r,b,g = image[:,:,0],image[:,:,1],image[:,:,2]
        avgs = [np.mean(r[r.nonzero()]),
                np.mean(b[b.nonzero()]),
                np.mean(g[g.nonzero()])]
        return avgs
    filtered_contours = []
    mask = np.zeros(image.shape)
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if (area > min_area) & (area < max_area):
            polygon = cv2.approxPolyDP(
                contour,0.01 * cv2.arcLength(contour,True),True
            )
            if (len(polygon) > min_poly):
                cv2.drawContours(mask,contours,i,color = (1,1,1),thickness = -1)
                avgs = masked_mean(image,mask)
                cv2.drawContours(mask,contours,i,color = (0,0,0),thickness = -1)
                if avgs[0] / avgs[1] > 1.2:
                    filtered_contours.append(contour)
    return filtered_contours

def filter_contours_size_color_2(contours,min_poly,min_area,max_area,image):
    """
    A function to perform both filterings in a 'single' loop.
    """
    def masked_mean(image,mask):
        return cv2.mean(image,mask = mask)
    filtered_contours = []
    for i in range(len(contours)):
        contour = contours[i].astype('float32')
        area = cv2.contourArea(contour)
        convex_hull = cv2.convexHull(contour)
        if (area > min_area) & (area < max_area):
            polygon = cv2.approxPolyDP(
                contour,0.01 * cv2.arcLength(contour,True),True
            )
            if (len(polygon) > min_poly):
                mask = np.zeros(image[:,:,0].shape,np.uint8)
                cv2.drawContours(mask,contours,i,color = (1,1,1),thickness = -1)
                avgs = masked_mean(image,mask)
                if avgs[0] > 170 and np.all(np.array(avgs) < 220):
                    filtered_contours.append(contour.astype('int32'))
    return filtered_contours

def watershed_contours(contours,image):
    shape = image.shape[:2]
    kernel = np.ones((3,3))
    binary_image = np.zeros(shape,np.uint8)
    cv2.drawContours(binary_image,contours,-1,255,cv2.FILLED)

    D = ndimage.distance_transform_edt(binary_image)
    binary_image = cv2.morphologyEx(binary_image,cv2.MORPH_OPEN,kernel)

    localMax = peak_local_max(D, indices=False,
                              min_distance=15,labels=binary_image)
    markers = ndimage.label(localMax, structure=np.ones((3,3)))[0]
    labels = watershed(-D, markers,
                       mask=binary_image,
                       connectivity=4)
    mask = np.ones(binary_image.shape, dtype=np.float32) * -1
    for label in np.unique(labels)[1:]:
        mask[labels == label] = 255
    mask[mask == -1] = 0
    mask = np.uint8(mask)
    contours = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    return contours

def fit_ellipse(contours):
    ellipses = []
    for contour in contours:
        if contour.shape[0] > 10:
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
    return ellipses

def filter_ellipse_size(ellipses,contours,ab_ratio,min_area,max_area):
    filtered_ellipses = []
    filtered_contours = []
    for i in range(len(ellipses)):
        ellipse = ellipses[i]
        a,b = ellipse[1][0]/2, ellipse[1][1]/2
        area = a * b * pi
        ab_condition = (a/b < ab_ratio) & (b/a < ab_ratio)
        area_condition = (area > min_area) & (area < max_area)
        if ab_condition * area_condition:
            filtered_ellipses.append(ellipse)
            filtered_contours.append(contours[i])
    return filtered_ellipses,filtered_contours

def get_ellipse_residuals(contour,ellipse):
    contour = np.squeeze(contour)
    ellipse_model = EllipseModel()
    ellipse_model.params = [ellipse[0][0],ellipse[0][1],
                            ellipse[1][0],ellipse[1][1],
                            ellipse[2]]
    return ellipse_model.residuals(contour)

def get_ellipse_area(ellipses):
    areas = []
    for ellipse in ellipses:
        a,b = ellipse[1]
        area = (a * b * pi) / 4
        areas.append(area)
    return areas

def draw_ellipses(ellipses,image):
    image = image.copy()
    for ellipse in ellipses:
        cv2.ellipse(image,ellipse,(0,255,0),2)
    return image

def wraper(image):
    image[image < 0] = 0
    edge_image = get_edges(image)
    contours = get_contours(edge_image)
    contour_image = np.zeros_like(image)
    cv2.drawContours(image=contour_image,
                     contours=contours,
                     contourIdx=-1,
                     color=(255,255,255),
                     thickness=-1)
    contour_image = contour_image[:,:,0] - edge_image
    contours = get_contours(contour_image)
    contours_filtered = filter_contours_size_color_2(contours,8,300,2000,image)
    contours_filtered = watershed_contours(contours_filtered,image)
    fitted_ellipses = fit_ellipse(contours_filtered)
    filtered_ellipses,filtered_contours = filter_ellipse_size(
        fitted_ellipses,contours_filtered,1.5,300,1500)
    contour_image = np.zeros_like(image)
    cv2.drawContours(image=contour_image,
                     contours=filtered_contours,
                     contourIdx=-1,
                     color=(255,255,255),
                     thickness=-1)
    return image,contour_image

def wraper(image,rescale_factor=1):
    image[image < 0] = 0
    edge_image = get_edges(image)
    contours = get_contours(edge_image)
    contour_image = np.zeros_like(image)
    cv2.drawContours(image=contour_image,
                     contours=contours,
                     contourIdx=-1,
                     color=(255,255,255),
                     thickness=-1)
    contour_image = contour_image[:,:,0] - edge_image
    contours = get_contours(contour_image)
    contours_filtered = filter_contours_size_color_2(
        contours,8,300/rescale_factor,2000/rescale_factor,image)
    fitted_ellipses = fit_ellipse(contours_filtered)
    filtered_ellipses,filtered_contours = filter_ellipse_size(
        fitted_ellipses,contours_filtered,1.5,
        300/rescale_factor,1500/rescale_factor)
    contour_image = np.zeros_like(image)
    cv2.drawContours(image=contour_image,
                     contours=filtered_contours,
                     contourIdx=-1,
                     color=(255,255,255),
                     thickness=-1)
    return image,contour_image

def otsu_threshold(im_array):
    """Transforms an image into an Otsu thresholded image."""
    if len(im_array.shape) == 3:
        im_array = np.mean(im_array.copy(),axis = 2)
    otsu_image = np.zeros(im_array.shape)
    otsu_value = filters.threshold_otsu(im_array)
    otsu_image[im_array > otsu_value] = 255
    return otsu_image

if __name__ == '__main__':
    IMAGE = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]
    output = wraper(imread(IMAGE))

    image = output[0]
    contour_image = output[1]

    Image.fromarray(contour_image).save(OUTPUT_PATH)
