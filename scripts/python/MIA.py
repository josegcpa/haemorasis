"""
Convenience script for masked image analysis.
"""

from masked_image_analysis import wrapper
from masked_image_analysis import wrapper_single_image
from masked_image_analysis import wrapper_single_image_separate

MIA_FEATURES = [
    'eccentricity','area','perimeter',
    'circle_variance','ellipse_variance',
    'convexity','solidity',
    'cdf_mean','cdf_std','cdf_max','cdf_min',
    'cuf_mean','cuf_std','cuf_max','cuf_min',
    'cdf_noiseless_moment_0','cdf_noiseless_moment_1',
    'cdf_noiseless_moment_2',
    'invariant_region_moments_0','invariant_region_moments_1',
    'invariant_region_moments_2','invariant_region_moments_3',
    'invariant_region_moments_4','invariant_region_moments_5',
    'invariant_region_moments_6',
    'cdf_fc',
    'peak_profile_major_fc','peak_profile_major_std',
    'peak_profile_major_max','peak_profile_major_min',
    'peak_profile_minor_fc','peak_profile_minor_std',
    'peak_profile_minor_max','peak_profile_minor_min',
    'mass_displacement_red','mass_displacement_green',
    'mass_displacement_blue','mass_displacement_av',
    'contrast','energy','homogeneity','correlation']

MIA_FEATURES_NUCLEUS = [
    "eccentricity","area_separate","perimeter_separate",
    "circle_variance","ellipse_variance",
    "convexity_separate","solidity_separate",
    "mass_displacement_red","mass_displacement_green",
    "mass_displacement_blue","mass_displacement_av"]