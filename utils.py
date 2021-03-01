"""
Utility functions for data loading / processing
March 2020
@author: Cian Scannell - cian.scannell@kcl.ac.uk
"""

import os
import csv
import random
import numpy as np
import nibabel as nib 
from glob import glob
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize

from skimage.measure import label
# from skimage.exposure import equalize_adapthist

domain_info = {}
with open('M&Ms Dataset Information.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in readCSV:
        if count:
            if row[1] == 'A':
                domain_info[row[0]] = 0
            if row[1] == 'B':
                domain_info[row[0]] = 1
            if row[1] == 'C':
                domain_info[row[0]] = 2
        count = 1

def get_domain_data(the_dir, no_domains, labelled=True, train=True):
    """
    Generator function which loads NIfTI images and correspoding masks (if they exist)
    from a given directory and optional shuffles the order
    :param the_dir: string, the path to the directory containing the images
    :param train: bool, if True to shuffle the order of the files and if False to pad images for prediction
    """

    im_files = sorted(glob(os.path.join(the_dir,'*sa.nii.gz')))
    seg_files = sorted(glob(os.path.join(the_dir,'*sa_gt.nii.gz')))
    if len(seg_files) == 0: # checking for the segmentations    
        # since there is no segmentations just use the images themselves as placeholders
        seg_files = sorted(glob(os.path.join(the_dir,'*sa.nii.gz')))
    im_seg_iterator = list(zip(im_files, seg_files))
    if train:
        random.shuffle(im_seg_iterator)
    for im_file,seg_file in im_seg_iterator:
        code = os.path.split(im_file)[1][:-10] 
        if labelled:
            x_nib, y_nib = nib.load(im_file), nib.load(seg_file)
            x = x_nib.get_fdata()
            y = y_nib.get_fdata()
            rx = x_nib.header['pixdim'][1]/1.25
            ry = x_nib.header['pixdim'][2]/1.25
            x = (( x - np.amin(x) ) / ( np.amax(x) - np.amin(x) ))[...,np.unique(np.where(y>0)[3])]
            y = y[...,np.unique(np.where(y>0)[3])]

            rescale_x = np.zeros((int(rx*x.shape[0]), int(ry*x.shape[1]), x.shape[2], x.shape[3]))
            rescale_y = np.zeros((int(rx*y.shape[0]), int(ry*y.shape[1]), y.shape[2], y.shape[3]))
            for s1 in range(x.shape[2]):
                for s2 in range(x.shape[3]):
                    rescale_x[...,s1,s2] = resize(x[...,s1,s2],(int(rx*x.shape[0]), int(ry*x.shape[1])))
                    rescale_y[...,s1,s2] = resize(y[...,s1,s2],(int(rx*y.shape[0]), int(ry*y.shape[1])),order=0)
            yield rescale_x, rescale_y, to_categorical(domain_info[code],num_classes=no_domains)   
        else:
            x = nib.load(im_file).get_fdata()
            yield (( x - np.amin(x) ) / ( np.amax(x) - np.amin(x) )), (( x - np.amin(x) ) / ( np.amax(x) - np.amin(x) )), to_categorical(domain_info[code],num_classes=no_domains)

def get_domain_data_only(the_dir, no_domains, train=True):
    """
    Generator function which loads NIfTI images and correspoding masks (if they exist)
    from a given directory and optional shuffles the order
    :param the_dir: string, the path to the directory containing the images
    :param train: bool, if True to shuffle the order of the files and if False to pad images for prediction
    """

    im_files = sorted(glob(os.path.join(the_dir,'*sa.nii.gz')))
    seg_files = sorted(glob(os.path.join(the_dir,'*sa_gt.nii.gz')))
    if len(seg_files) == 0: # checking for the segmentations    
        # since there is no segmentations just use the images themselves as placeholders
        seg_files = sorted(glob(os.path.join(the_dir,'*sa.nii.gz')))
    im_seg_iterator = list(zip(im_files, seg_files))
    if train:
        random.shuffle(im_seg_iterator)
    for im_file,seg_file in im_seg_iterator:
        code = os.path.split(im_file)[1][:-10] 
        x_nib = nib.load(im_file)
        x = x_nib.get_fdata()
        rx = x_nib.header['pixdim'][1]/1.25
        ry = x_nib.header['pixdim'][2]/1.25
        rint = [np.random.randint(x.shape[-1]),np.random.randint(x.shape[-1])]
        x = (( x - np.amin(x) ) / ( np.amax(x) - np.amin(x) ))[...,rint]
        rescale_x = np.zeros((int(rx*x.shape[0]), int(ry*x.shape[1]), x.shape[2], x.shape[3]))
        for s1 in range(x.shape[2]):
            for s2 in range(x.shape[3]):
                rescale_x[...,s1,s2] = resize(x[...,s1,s2],(int(rx*x.shape[0]), int(ry*x.shape[1])))

        yield rescale_x, to_categorical(domain_info[code],num_classes=no_domains)
