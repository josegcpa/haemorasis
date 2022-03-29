"""
Set of functions to characterise shape.
"""

from math import pi
import numpy as np
import time
from scipy.stats import moment
from scipy.interpolate import interp1d
import cv2

# Shape parameters

def axis_of_least_inertia(x,y):
    """
    Calculates the axis of least inertia.

    Args:
        * x - array with all centered x coordinates belonging to the object
        * y - array with all centered y coordinates belonging to the object

    Returns:
        * theta - the angle of the axis of least inertia
    """

    a = np.sum(np.square(x))
    b = 2 * np.sum(x * y)
    c = np.sum(np.square(y))

    alpha = 0.5 * np.arctan(b / (a - c))
    di_dalpha = 2 * (a - c) * np.cos(2 * alpha) + 2 * b * np.sin(2 * alpha)

    if di_dalpha < 0:
        theta = alpha + pi/2
    else:
        theta = alpha

    return theta

def rotate(x,y,theta):
    """
    Rotates x and y coordinates over an angle theta.

    Args:
        * x - array with x coordinates
        * y - array with y coordinates
        * theta - rotation angle

    Returns:
        * new_x - rotated x coordinates
        * new_y - rotated y coordinates
    """
    gx,gy = centroid(x,y)
    x_center = x - gx
    y_center = y - gy
    new_x = x_center * np.cos(theta) + y_center * np.sin(theta) + gx
    new_y = y_center * np.cos(theta) - x_center * np.sin(theta) + gy

    return new_x,new_y

def major_axis_skeleton(x,y,n_splits=50):
    """
    Calculates the values that exist along the skeleton as deviation from the
    central line. The problem with this method (and other similar methods) is
    that it assumes the object is not very curved.
    """
    gx,gy = sa.centroid(x,y)
    theta = sa.axis_of_least_inertia(new_x - gx,new_y - gy)
    rotated_x,rotated_y = sa.rotate(new_x,new_y,theta)
    coords = np.arange(np.min(rotated_x),np.max(rotated_x),
                       dtype=np.uint16)
    splits = np.array_split(coords,50)
    skeleton_x, skeleton_y = [],[]
    for split in splits:
        tmp_x = np.uint16(np.mean(split)) - gx
        tmp_y = np.uint16(np.mean(np.where(tmp[split,:] > 0)[1])) - gy
        skeleton_x.append(tmp_x)
        skeleton_y.append(tmp_y)
    rot_skel_x,rot_skel_y = rotate(skeleton_x,skeleton_y,-theta)
    return rot_skel_x,rot_skel_y

def centroid(x,y):
    """
    Calculates the centroid for a given set of x and y coordinates.

    Args:
        * x - array with all x coordinates belonging to the object
        * y - array with all y coordinates belonging to the object

    Returns:
        * gx - x coordinate for the centroid
        * gy - y coordinate for the centroid
    """
    gx = np.mean(x)
    gy = np.mean(y)

    return gx,gy

def correlation(x,y,gx,gy):
    """
    Calculates the correlation for a given set of x and y coordinates and their
    centroid.

    Args:
        * x - array with all x coordinates belonging to the object
        * y - array with all y coordinates belonging to the object
        * gx - x coordinate for the centroid
        * gy - y coordinate for the centroid

    Returns:
        * cxx - correlation between x and gx
        * cyy - correlation between y and gy
        * cxy - correlation between x,y and gx,gy
    """
    cxx = np.mean(np.square(x - gx))
    cyy = np.mean(np.square(y - gy))
    cxy = np.mean((x - gx) * (y - gy))
    return cxx,cyy,cxy

def eccentricity(cxx,cyy,cxy):
    """
    Calculates the eccentricity (ratio between major and minor axes).

    Args:
        * cxx - correlation between x and gx
        * cyy - correlation between y and gy
        * cxy - correlation between x,y and gx,gy

    Returns:
        * eccentricity
    """
    lambda_1 = cxx + cyy + np.sqrt((cxx + cyy)**2 - 4 * (cxx * cyy - cxy**2))
    lambda_2 = cxx + cyy - np.sqrt((cxx + cyy)**2 - 4 * (cxx * cyy - cxy**2))

    return lambda_2/lambda_1

def cnt_to_xy(cnt):
    """
    Converts a contour as returned by opencv into x and y coordinates for the
    boundary of the object.

    Args:
        * cnt - contour, as returned by cv2.findContours

    Returns:
        * x - x coordinates for the boundary
        * y - y coordinates for the boundary
    """
    return cnt[:,:,0],cnt[:,:,1]

def area(cnt):
    """
    Calculates the area of a contour. Wrapper for opencv function.

    Args:
        * cnt - contour, as returned by cv2.findContours

    Returns:
        * area
    """
    return cv2.contourArea(cnt)

def area_separate(cnt):
    """
    Calculates the total area of a list of contours.

    Args:
        * cnt - list of contours, as returned by cv2.findContours

    Returns:
        * total area
    """
    return sum([cv2.contourArea(x) for x in cnt])

def perimeter(cnt):
    """
    Calculates the perimeter of a contour. Wrapper for opencv function.

    Args:
        * cnt - contour, as returned by cv2.findContours

    Returns:
        * perimeter
    """
    return cv2.arcLength(cnt,True)

def perimeter_separate(cnt):
    """
    Calculates the total perimeter of a list of contours.

    Args:
        * cnt - list of contours

    Returns:
        * total perimeter
    """
    return sum([cv2.arcLength(x,True) for x in cnt])

def circle_variance(cnt):
    """
    Calculates the circle variance of a contour (how "different" a contour is
    from the circle fitted to its contour).

    Args:
        * cnt - contour, as returned by cv2.findContours

    Returns:
        * circle variance
    """
    return (area(cnt) ** 2) / (perimeter(cnt) ** 2)

def ellipse_variance(x_bound,y_bound,gx,gy,cxx,cyy,cxy):
    """
    Calculates the ellipse variance of a contour (how "different" a contour is
    from the ellipse fitted to its contour, assuming the ellipse has the same
    covariance as the object).

    Args:
        * x_bound - coordinates belonging to the boundary of x
        * y_bound - coordinates belonging to the boundary of x
        * gx - x coordinate for the centroid
        * gy - y coordinate for the centroid
        * cxx - correlation between x and gx
        * cyy - correlation between y and gy
        * cxy - correlation between x,y and gx,gy

    Returns:
        * e_va - ellipse variance
    """

    vi = np.squeeze(np.stack([x_bound - gx,y_bound - gy]),axis=2)
    c = np.array([[cxx,cxy],[cxy,cyy]])
    c_inv = np.linalg.inv(c)
    di = np.sqrt(np.matmul(np.matmul(np.transpose(vi),c_inv),vi))
    di = di[~np.isnan(di)]
    mu_r = np.mean(di)
    std_r = np.std(di)

    e_va = std_r / mu_r

    return e_va

def convex_hull(cnt):
    """
    Calculates the convex hull of a contour. Wrapper for opencv function.

    Args:
        * cnt - contour, as returned by cv2.findContours

    Returns:
        * convex hull
    """
    return cv2.convexHull(cnt)

def convexity(cnt,hull):
    """
    Calculates the convexity a contour (ratio between the convex hull and the
    objects perimeters).

    Args:
        * cnt - contour, as returned by cv2.findContours

    Returns:
        * convexity
    """
    return perimeter(hull) / perimeter(cnt)

def convexity_separate(cnt,hull):
    """
    Calculates the convexity for a list of contours.

    Args:
        * cnt - list of contours, as returned by cv2.findContours

    Returns:
        * convexity
    """
    return perimeter(hull) / perimeter_separate(cnt)

def solidity(cnt,hull):
    """
    Calculates the solidity a contour (ratio between the convex hull and the
    objects areas).

    Args:
        * cnt - contour, as returned by cv2.findContours

    Returns:
        * solidity
    """
    return area(hull) / area(cnt)

def solidity_separate(cnt,hull):
    """
    Calculates the solidity for a list of contours.

    Args:
        * cnt - list of contours, as returned by cv2.findContours

    Returns:
        * solidity
    """
    return area(hull) / area_separate(cnt)

# One-dimensional function for shape representation

def centroid_distance_function(x_bound,y_bound,gx,gy):
    """
    Calculates the centroid distance function (array containing the distance
    between all points in the contour and the centroid).

    Args:
        * x_bound - coordinates belonging to the boundary of x
        * y_bound - coordinates belonging to the boundary of x
        * gx - x coordinate for the centroid
        * gy - y coordinate for the centroid

    Returns:
        * centroid distance function
    """
    return np.sqrt(np.square(x_bound - gx) + np.square(y_bound - gy))

def tangent_angle(x_bound,y_bound):
    """
    Calculates the tangent angle (angle at each point of the contour).

    Args:
        * x_bound - coordinates belonging to the boundary of x
        * y_bound - coordinates belonging to the boundary of x

    Returns:
        * tangent angles
    """
    x_bound_2 = np.concatenate([x_bound[:-1],x_bound[:-1]])
    y_bound_2 = np.concatenate([y_bound[:-1],y_bound[:-1]])

    return np.arctan((y_bound - y_bound_2) / (x_bound - x_bound_2))

def derivative(vector):
    """
    Calculates the derivative at each point of a vector.
    Careful: this assumes a closed contour.

    Args:
        * vector - vector belonging to a coordinate of a closed shape

    Returns:
        * derivative
    """
    vector_nm1 = np.concatenate([np.array([vector[-1]]),vector[:-1]])
    return vector - vector_nm1

def curvature(x_bound,y_bound):
    """
    Calculates the curvature of a contour provided its x and y coordinates. If
    the size of the curve is relevant, k should be used. If this is considered
    to be irrelevant, than scale_inv_k should be used.

    Args:
        * x_bound - coordinates belonging to the boundary of x
        * y_bound - coordinates belonging to the boundary of x

    Returns:
        * k - curvature
        * scale_inv_k - curvature devided by the curvature mean (scale
        invariant)
    """
    x_prime = derivative(x_bound)
    y_prime = derivative(y_bound)
    x_2prime = derivative(x_prime)
    y_2prime = derivative(y_prime)

    k = np.divide(
        x_prime * y_2prime - y_prime * x_2prime,
        np.power(np.square(x_prime) + np.square(y_prime),1.5))

    scale_inv_k = k / np.mean(k)
    return k,scale_inv_k

def triangle_area(x,y):
    """
    Calculates the area of a triangle given the x and y coordinates for 3
    points.

    Args:
        * x - 3 x coordinates for the triangle
        * y - 3 y coordinates for the triangle

    Returns:
        * area
    """
    coord_array = np.array(
        [[x[0],y[0],1],
         [x[1],y[1],1],
         [x[2],y[2],1]])
    return 0.5 * np.abs(np.linalg.det(coord_array))

def area_function(x_bound,y_bound,gx,gy):
    """
    Calculates the area function for a contour. This considers triangles
    between two consecutive points and the centroid.

    Args:
        * x_bound - coordinates belonging to the boundary of x
        * y_bound - coordinates belonging to the boundary of x
        * gx - x coordinate for the centroid
        * gy - y coordinate for the centroid

    Returns:
        * areas - vector containing all areas
    """
    x_bound_2 = np.concatenate([x_bound[1:],np.array([x_bound[0]])])
    y_bound_2 = np.concatenate([y_bound[1:],np.array([y_bound[0]])])
    areas = []
    for x1,y1,x2,y2 in zip(x_bound,y_bound,x_bound_2,y_bound_2):
        areas.append(triangle_area([x1[0],x2[0],gx],[y1[0],y2[0],gy]))
    return areas

def triangle_area_representation(x_bound,y_bound):
    """
    Calculates the triangle area representation for a contour. This considers
    triangles between three adjacent points.

    Args:
        * x_bound - coordinates belonging to the boundary of x
        * y_bound - coordinates belonging to the boundary of x

    Returns:
        * areas - vector containing all areas
    """
    x_bound_0 = np.concatenate([np.array([x_bound[-1]]),x_bound[:-1]])
    y_bound_0 = np.concatenate([np.array([y_bound[-1]]),y_bound[:-1]])
    x_bound_2 = np.concatenate([x_bound[1:],np.array([x_bound[0]])])
    y_bound_2 = np.concatenate([y_bound[1:],np.array([y_bound[0]])])

    areas = []
    for x1,y1,x2,y2,x3,y3 in zip(x_bound_0,y_bound_0,
                                 x_bound,y_bound,
                                 x_bound_2,y_bound_2):
        areas.append(triangle_area([x1[0],x2[0],x3[0]],[y1[0],y2[0],y3[0]]))
    return areas

# Moments

def moments(z,central=False,normalized=False,m=1):
    if central == False:
        moment = np.mean(np.power(z,m))
    else:
        moment = np.mean(np.power(z - np.mean(z),m))

    if normalized == True:
        moment = moment / np.power(np.std(z),m/2)

    return moment

def noiseless_moments(z):
    m_1 = moments(z)
    mu_2 = moments(z,central=True,m=2)
    mu_3 = moments(z,central=True,m=3)
    mu_4 = moments(z,central=True,m=4)
    return [np.sqrt(mu_2)/m_1,
            mu_3/np.power(mu_2,1.5),
            mu_4/np.square(mu_2)]

def region_moments(x,y,p,q,centered=False):
    if centered == True:
        return np.sum(
            np.matmul(
                np.power(np.expand_dims(x - np.mean(x),1),p),
                np.power(np.expand_dims(y - np.mean(y),0),q)
                )
            )
    else:
        return np.sum(
            np.matmul(
                np.power(np.expand_dims(x,1),p),
                np.power(np.expand_dims(y,0),q)
                )
            )

def invariant_region_moments(x,y):
    mu = region_moments(x,y,0,0,centered=True)
    mu_20 = region_moments(x,y,2,0,centered=True)/(mu**2)
    mu_02 = region_moments(x,y,0,2,centered=True)/(mu**2)
    mu_11 = region_moments(x,y,1,1,centered=True)/(mu**2)
    mu_30 = region_moments(x,y,3,0,centered=True)/(mu**2.5)
    mu_03 = region_moments(x,y,0,3,centered=True)/(mu**2.5)
    mu_21 = region_moments(x,y,2,1,centered=True)/(mu**2.5)
    mu_12 = region_moments(x,y,1,2,centered=True)/(mu**2.5)

    phi_1 = mu_20 + mu_02
    phi_2 = (mu_20 - mu_02) ** 2 + 4 * mu_11 * mu_11
    phi_3 = (mu_30 + 3 * mu_12) ** 2 + (3 * mu_21 + mu_03) ** 2
    phi_4 = (mu_30 + mu_12) ** 2 + (mu_21 + mu_03) ** 2
    phi_5 = (mu_30-3*mu_12) * (mu_30+mu_12) * ((mu_30+mu_12)**2-3*(mu_21+mu_03)**2) + (3*mu_21-mu_03) * (mu_21+mu_03) * (3*(mu_30+mu_12)**2-(mu_21+mu_03)**2)
    phi_6 = (mu_20-mu_02) * ((mu_30+mu_12)**2-(mu_21+mu_03)**2) + 4*mu_11*mu_11*(mu_30+mu_12)*(mu_21+mu_03)
    phi_7 = (3*mu_21-mu_03)*(mu_30+mu_12)*((mu_30 + mu_12)**2 - 3*(mu_21+mu_03)**2) + (3*mu_12-mu_03)*(mu_21+mu_03) * (3*(mu_30+mu_12)**2-(mu_21+mu_03)**2)

    return [phi_1,phi_2,phi_3,phi_4,phi_5,phi_6,phi_7]
# Shape space transforms

def undersample(z,n=20):
    """
    Undersamples a vector z to have n elements.

    Args:
        * z - vector
        * n - number of elements in the output vector

    Returns:
        * undersampled_z - undersampled vector
    """

    if z.shape[0] >= n:
        undersampled_z = [np.mean(x) for x in np.array_split(z,n)]
        return undersampled_z
    else:
        return None

def inv_fft(z):
    """
    Calculates the scale and rotation invariant Fourier coefficients of a
    vector.

    Args:
        * z - vector

    Returns:
        * rot_scale_inv_fd - rotation and scale invariant Fourier descriptors.
    """

    a = np.fft.fft(z)
    b = a / a[1]
    rot_scale_inv_fd = np.abs(b)
    return rot_scale_inv_fd

def fourier_complexity(z):
    """
    Calculate the area under the curve that describes the reconstruction error
    from a different number of components.

    Args:
        * z - vector

    Returns:
        * fourier_complexity - a value that represents the complexity of the
        signal as the area under the reconstruction error curve.
    """

    signal = (z - np.mean(z))/np.std(z)
    fft = np.fft.fft(signal)
    signal_t = signal.T
    error_list = []
    for i in range(1,fft.shape[0]):
        s = fft[:i]
        error_list.append(np.abs(np.fft.ifft(s,n=signal.shape[0]) - signal_t).mean())

    return np.trapz(error_list) / len(error_list)

def subsample(y,n=150):
    if y.shape[0] < n:
        return y
    a = time.time()
    x = np.linspace(0,1,num=y.shape[0],endpoint=False)
    new_x = np.linspace(0,1,num=n,endpoint=False)
    return interp1d(x,y)(new_x)

def wrapper(x,y,cnt):
    gx,gy = centroid(x,y)
    x_bound,y_bound = cnt_to_xy(cnt)
    hull = convex_hull(cnt)
    cxx,cyy,cxy = correlation(x,y,gx,gy)
    out = {}
    cdf = centroid_distance_function(x_bound,y_bound,gx,gy)
    cuf = curvature(x_bound,y_bound)[1]
    cdf = subsample(cdf[:,0])[:,np.newaxis]
    cuf = subsample(cuf[:,0])[:,np.newaxis]

    out['eccentricity'] = eccentricity(cxx,cyy,cxy)
    out['area'] = area(cnt)
    out['perimeter'] = perimeter(cnt)
    out['circle_variance'] = (out['area'] ** 2) / (out['perimeter'] ** 2)
    out['ellipse_variance'] = ellipse_variance(
        x_bound,y_bound,gx,gy,cxx,cyy,cxy)
    out['convexity'] = convexity(cnt,hull)
    out['solidity'] = solidity(cnt,hull)

    out['cdf_mean'] = np.mean(cdf)
    out['cdf_std'] = np.std(cdf)
    out['cdf_max'] = np.max(cdf)
    out['cdf_min'] = np.min(cdf)

    out['cuf_mean'] = np.mean(cuf)
    out['cuf_std'] = np.std(cuf)
    out['cuf_max'] = np.max(cuf)
    out['cuf_min'] = np.min(cuf)

    nm = noiseless_moments(cdf)
    for i,m in enumerate(nm):
        out['cdf_noiseless_moment_{}'.format(i)] = m
    irm = invariant_region_moments(x,y)
    for i,m in enumerate(irm):
        out['invariant_region_moments_{}'.format(i)] = m

    if len(cdf) >= 20:
        out['cdf_fc'] = fourier_complexity(cdf)
    else:
        out['cdf_fc'] = np.nan

    return out

def wrapper_separate(x,y,cnt):
    out = {}
    gx,gy = centroid(x,y)
    cnt_compiled = np.concatenate(cnt,axis=0)
    x_bound,y_bound = cnt_to_xy(cnt_compiled)
    hull = convex_hull(np.concatenate(cnt,axis=0))
    cxx,cyy,cxy = correlation(x,y,gx,gy)

    out['eccentricity'] = eccentricity(cxx,cyy,cxy)
    out['area_separate'] = area_separate(cnt)
    out['perimeter_separate'] = perimeter_separate(cnt)
    out['circle_variance'] = (out['area_separate'] ** 2) / (out['perimeter_separate'] ** 2)
    out['ellipse_variance'] = ellipse_variance(
        x_bound,y_bound,gx,gy,cxx,cyy,cxy)
    out['convexity_separate'] = convexity_separate(cnt,hull)
    out['solidity_separate'] = solidity_separate(cnt,hull)
    return out

ALL_FUNCTIONS = {
    'area':area,
    'area_function':area_function,
    'axis_of_least_inertia':axis_of_least_inertia,
    'centroid':centroid,
    'centroid_distance_function':centroid_distance_function,
    'circle_variance':circle_variance,
    'cnt_to_xy':cnt_to_xy,
    'convex_hull':convex_hull,
    'convexity':convexity,
    'correlation':correlation,
    'curvature':curvature,
    'derivative':derivative,
    'eccentricity':eccentricity,
    'ellipse_variance':ellipse_variance,
    'fourier_complexity':fourier_complexity,
    'inv_fft':inv_fft,
    'invariant_region_moments':invariant_region_moments,
    'major_axis_skeleton':major_axis_skeleton,
    'moments':moments,
    'noiseless_moments':noiseless_moments,
    'perimeter':perimeter,
    'region_moments':region_moments,
    'rotate':rotate,
    'solidity':solidity,
    'tangent_angle':tangent_angle,
    'triangle_area':triangle_area,
    'triangle_area_representation':triangle_area_representation,
    'undersample':undersample
    }
