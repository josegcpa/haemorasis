"""
Functions to analyse the colour distribution of objects
"""

import numpy as np
import cv2
from shape_analysis import undersample,inv_fft,fourier_complexity

CONVOLUTION_OPERATOR = np.ones([3]) / 3

def peak_profiles(image,cnt):
    def floorint(arr):
        return np.int16(np.floor(arr))

    def size_cap(x,y):
        if x >= image.shape[0]:
            x = image.shape[0] < 1
        if y >= image.shape[1]:
            y = image.shape[1] < 1
        return x,y

    if (cnt.shape[0] > 10) * (len(cnt.shape) == 3):
        rect = cv2.minAreaRect(cnt)
        points = cv2.boxPoints(rect)
        y1,x1 = points[0]
        y2,x2 = points[1]
        y3,x3 = points[2]
        y4,x4 = points[3]
        mp1 = size_cap((x2+x1)/2,(y2+y1)/2)
        mp2 = size_cap((x3+x2)/2,(y3+y2)/2)
        mp3 = size_cap((x4+x3)/2,(y4+y3)/2)
        mp4 = size_cap((x1+x4)/2,(y1+y4)/2)

        dist_1 = np.sqrt((mp1[0] - mp3[0]) ** 2 + (mp1[1] - mp3[1]) ** 2)
        dist_2 = np.sqrt((mp2[0] - mp4[0]) ** 2 + (mp2[1] - mp4[1]) ** 2)
        dist_1 = np.ceil(dist_1).astype(np.int16)
        dist_2 = np.ceil(dist_2).astype(np.int16)

        x_coords_1 = floorint(np.linspace(mp1[0],mp3[0],dist_1))
        y_coords_1 = floorint(np.linspace(mp1[1],mp3[1],dist_1))
        x_coords_2 = floorint(np.linspace(mp2[0],mp4[0],dist_2))
        y_coords_2 = floorint(np.linspace(mp2[1],mp4[1],dist_2))

        color_distribution_list_1 = get_peak_profiles(
            image,x_coords_1,y_coords_1)

        color_distribution_list_2 = get_peak_profiles(
            image,x_coords_2,y_coords_2)

    else:
        color_distribution_list_1 = [np.nan,np.nan,np.nan,np.nan]
        color_distribution_list_2 = [np.nan,np.nan,np.nan,np.nan]

    return color_distribution_list_1,color_distribution_list_2

def get_peak_profiles(image,x_profile_coords,y_profile_coords):
    if len(x_profile_coords) >= 10:
        profile = image[x_profile_coords,y_profile_coords]
        profile = np.mean(profile,axis=1)
        profile = np.convolve(profile,CONVOLUTION_OPERATOR,mode='same')
        profile = profile - np.mean(profile)
        output = [
            fourier_complexity(profile),
            np.std(profile),
            np.max(profile),
            np.min(profile)
            ]

    else:
        output = [
            np.nan,
            np.nan,
            np.nan,
            np.nan
            ]

    return output

def center_of_mass(coords,values):
    def cm(values,coord):
        out = np.sum(values * coord) / np.sum(values)
        return out
    centers = [cm(values,coord) for coord in coords]
    return np.array(centers)

def mass_displacement(image,x,y):
    r_mass = image[x,y,0]
    b_mass = image[x,y,1]
    g_mass = image[x,y,2]
    av_mass = (r_mass + b_mass + g_mass) / 3.
    binary_mass = np.ones_like(av_mass)
    r_center = center_of_mass([x,y],r_mass)
    b_center = center_of_mass([x,y],b_mass)
    g_center = center_of_mass([x,y],g_mass)
    av_center = center_of_mass([x,y],av_mass)
    binary_center = center_of_mass([x,y],binary_mass)

    mass_displacement_list = [
        np.sqrt(np.sum(np.square(r_center - binary_center))),
        np.sqrt(np.sum(np.square(g_center - binary_center))),
        np.sqrt(np.sum(np.square(b_center - binary_center))),
        np.sqrt(np.sum(np.square(av_center - binary_center)))
    ]

    return mass_displacement_list

def wrapper(image,cnt,x,y):
    out = {}
    pp_1,pp_2 = peak_profiles(image,cnt)
    pp_1_fc,pp_1_std,pp_1_max,pp_1_min = pp_1
    pp_2_fc,pp_2_std,pp_2_max,pp_2_min = pp_2
    out['peak_profile_major_fc'] = pp_1_fc
    out['peak_profile_major_std'] = pp_1_std
    out['peak_profile_major_max'] = pp_1_max
    out['peak_profile_major_min'] = pp_1_min

    out['peak_profile_minor_fc'] = pp_2_fc
    out['peak_profile_minor_std'] = pp_2_std
    out['peak_profile_minor_max'] = pp_2_max
    out['peak_profile_minor_min'] = pp_2_min

    md = mass_displacement(image,x,y)
    out['mass_displacement_red'] = md[0]
    out['mass_displacement_green'] = md[1]
    out['mass_displacement_blue'] = md[2]
    out['mass_displacement_av'] = md[3]

    return out

def wrapper_separate(image,cnt,x,y):
    out = {}
    
    md = mass_displacement(image,x,y)
    out['mass_displacement_red'] = md[0]
    out['mass_displacement_green'] = md[1]
    out['mass_displacement_blue'] = md[2]
    out['mass_displacement_av'] = md[3]

    return out
