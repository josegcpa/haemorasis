"""
Script to characterise WBC and RBC in a whole blood slide from
a segmented cells HDF5.

Usage:
    python3 characterise_cells.py --help
"""

import sys
import argparse
import os
import numpy as np
import time
import h5py
import xgboost
import pickle
import cv2
import openslide
from multiprocessing import Pool
from skimage.transform import rescale
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import MIA

MIA_FEATURES = MIA.MIA_FEATURES
MIA_FEATURES_W_NUCLEUS = MIA_FEATURES.copy()
MIA_FEATURES_W_NUCLEUS.extend(
    [x+'_nuclear' for x in MIA.MIA_FEATURES_NUCLEUS])

def segment_nucleus(image,mask):
    idxs = np.where(mask>0)
    pixel_values = image[idxs]
    pixel_values_ = np.float32(pixel_values)
    _, labels, (centers) = cv2.kmeans(
        pixel_values_, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        10, cv2.KMEANS_RANDOM_CENTERS)
    nucleus_center = np.argmin(np.mean(centers,axis=-1))

    label_mask = np.zeros_like(mask)
    label_mask[idxs] = labels[:,0] + 1
    label_mask = np.where(label_mask == (nucleus_center+1),1,0)
    label_mask = label_mask.astype(np.uint8)
    labels, stats = cv2.connectedComponentsWithStats(label_mask, 4)[1:3]
    largest_label = 1 + np.where(stats[1:, cv2.CC_STAT_AREA] > 100)[0]
    label_mask = np.where(np.isin(labels,largest_label),1,0).astype(np.uint8)
    cnt = cv2.findContours(label_mask,cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(label_mask, cnt, -1, 1, -1)
    return label_mask,cnt

class FeatureExtractor:
    def __init__(self,rescale_factor,cell_type):
        self.rescale_factor = rescale_factor
        self.cell_type = cell_type
        self.get_fn()

    def rescale_image_mask(self,image,mask,cnt=None):
        image = rescale(image,self.rescale_factor,
                        anti_aliasing=True,multichannel=True,
                        preserve_range=True,clip=False)
        mask = rescale(mask,self.rescale_factor,
                        anti_aliasing=True,order=0,
                        preserve_range=True,clip=False)
        mask = np.uint8(np.round(mask))
        contours,_ = cv2.findContours(mask,2,1)
        cnt = contours[-1]
        return image,mask,cnt

    def identity(self,image,mask,cnt):
        return image,mask,cnt

    def characterise(self,image,mask,cnt=None):
        return MIA.wrapper_single_image(image,mask,cnt)

    def characterise_w_nucleus(self,image,mask,cnt=None):
        features_cell = MIA.wrapper_single_image(image,mask,cnt)
        nucleus_mask,nucleus_cnt = segment_nucleus(image,mask)
        features_nucleus = MIA.wrapper_single_image_separate(
            image,nucleus_mask,nucleus_cnt)
        for k in features_nucleus:
            features_cell[k+'_nuclear'] = features_nucleus[k]
        return features_cell

    def get_fn(self):
        if args.rescale_factor != 1:
            self.preproc_fn = self.rescale_image_mask
        else:
            self.preproc_fn = self.identity
        if args.cell_type == 'wbc':
            self.extract_feature_fn = self.characterise_w_nucleus
        elif args.cell_type == 'rbc':
            self.extract_feature_fn = self.characterise

    def extract_features(self,image,mask,cnt):
        image,mask,cnt = self.preproc_fn(image,mask,cnt)
        try:
            output = self.extract_feature_fn(image,mask,cnt)
        except:
            output = None
        return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog = 'characterise_cells.py',
        description = 'characterises cells and derives summary statistics')

    parser.add_argument('--slide_path',dest='slide_path',
                        action='store',default=None,
                        help="Path to whole blood slide")
    parser.add_argument('--segmented_cells_path',dest='segmented_cells_path',
                        action='store',default=None,
                        help='Path to hdf5 file containing segmented cells')
    parser.add_argument('--cell_type',dest='cell_type',
                        action='store',default='WBC',
                        help='WBC or RBC')
    parser.add_argument('--output_path',dest='output_path',
                        default=None,action='store',
                        help='Output path for the hdf5 file.')
    parser.add_argument('--n_processes',dest='n_processes',
                        action='store',default=2,type=int,
                        help='Number of concurrent processes')
    parser.add_argument('--rescale_factor',dest='rescale_factor',
                        action='store',default=1,type=float,
                        help='Factor to resize cells (due to different mpp).')
    parser.add_argument('--standardizer_params',dest='standardizer_params',
                        action='store',default="scripts/python/rbc_scaler_params",
                        type=str,help='Path to standardizer parameters.')
    parser.add_argument('--xgboost_model',dest='xgboost_model',
                        action='store',default="scripts/python/rbc_xgb_model",
                        type=str,help='Path to XGBoost model (RBC only).')

    args = parser.parse_args()
    args.cell_type = args.cell_type.lower()

    feature_extractor = FeatureExtractor(args.rescale_factor,args.cell_type)
    OS = openslide.OpenSlide(args.slide_path)

    def fn(X,Y,cnt):
        x,y = np.min(X)-5,np.min(Y)-5
        h,w = np.max(X)-x+10,np.max(Y)-y+10
        image = np.array(OS.read_region((x,y),0,(h,w)))
        mask = np.zeros([w,h],dtype=np.uint8)
        mask[Y-y,X-x] = 1
        return feature_extractor.extract_features(image,mask,cnt)

    pool = Pool(args.n_processes)

    if args.cell_type == 'wbc':
        MIA_FEATURES = MIA_FEATURES_W_NUCLEUS

    all_features = []
    all_ks = []
    image_sets = []
    k_sets = []
    cell_centers = []
    cell_idx = 0
    with h5py.File(args.segmented_cells_path,'r') as F:
        all_keys = list(F.keys())
        for k in tqdm(all_keys):
            X = F[k]['X'][()]
            Y = F[k]['Y'][()]
            cnt = F[k]['cnt'][()]
            # x,y = np.min(X)-5,np.min(Y)-5
            # h,w = np.max(X)-x+10,np.max(Y)-y+10

            # image = np.array(OS.read_region((x,y),0,(h,w)))
            # mask = np.zeros([w,h],dtype=np.uint8)
            # mask[Y-y,X-x] = 1
            # image_sets.append([image,mask,cnt])
            image_sets.append([X,Y,cnt])
            k_sets.append(k)
            if len(image_sets) == args.n_processes:
                # output = pool.starmap(feature_extractor.extract_features,image_sets)
                output = pool.starmap(fn,image_sets)
                for element,k in zip(output,k_sets):
                    if element is not None:
                        cell_center = k[1:-1].split(',')
                        cx = float(cell_center[0])
                        cy = float(cell_center[0])
                        features = []
                        for f in MIA_FEATURES:
                            features.append(element[f])
                        features = np.array(features)
                        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                            pass
                        else:
                            all_ks.append(cell_idx)
                            all_features.append(features)
                            cell_centers.append(k)
                    cell_idx += 1
                image_sets = []
                k_sets = []

    if len(image_sets) > 0:
        output = pool.starmap(
            feature_extractor.extract_features,image_sets)
        for element,k in zip(output,k_sets):
            if element is not None:
                features = []
                for f in MIA_FEATURES:
                    features.append(element[f])
                features = np.array(features)
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    pass
                else:
                    all_ks.append(cell_idx)
                    all_features.append(features)
                    cell_centers.append(k)
            cell_idx += 1

    all_features = np.array(all_features)
    all_ks = np.array(all_ks)
    if args.cell_type == 'rbc':
        standardized_scaler = StandardScaler()
        fitted_model = xgboost.XGBClassifier(
            booster='dart',eval_metric="logloss")
        with open(args.standardizer_params,"rb") as o:
            sc_params = pickle.load(o)

        standardized_scaler.n_features_in_ = len(sc_params['mean'])
        standardized_scaler.mean_ = sc_params['mean']
        standardized_scaler.scale_ = sc_params['std']
        standardized_scaler.var_ = sc_params['std']**2
        fitted_model.load_model(args.xgboost_model)

        if all_features.size > 0:
            all_features_ = standardized_scaler.transform(all_features)
            predictions = fitted_model.predict(all_features_)
            all_features = all_features[predictions,:]
            all_ks = all_ks[predictions]

    with h5py.File(args.output_path,'w') as F_out:
        F_out['cells/0'] = all_features
        F_out['means'] = np.mean(all_features,axis=0)
        F_out['variances'] = np.var(all_features,axis=0)
        F_out['cell_centers'] = np.array(cell_centers,dtype="S")
        F_out['cell_center_idxs'] = all_ks

