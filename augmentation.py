import numpy as np 
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist

import gryds


def do_augmentation2(image, label, sz, ia=False):

    random_grid = np.random.rand(2, 7, 7)
    random_grid -= 0.5
    random_grid /= 12
    # Define a B-spline transformation object
    bspline_trf = gryds.BSplineTransformation(random_grid)

    # rotate between -pi/8 and pi/8
    rot = np.random.rand()*np.pi/4 - np.pi/8
    # scale between 0.9 and 1.1
    scale_x = np.random.rand()*0.2 + 0.9
    scale_y = np.random.rand()*0.2 + 0.9
    # translate between -10% and 10%
    trans_x = np.random.rand()*.2 - .1
    trans_y = np.random.rand()*.2 - .1

    affine_trf = gryds.AffineTransformation(
        ndim=2,
        angles=[rot], # the rotation angle
        scaling=[scale_x, scale_y], # the anisotropic scaling
        translation=[trans_x, trans_y], # translation
        center=[0.5, 0.5] # center of rotation
    )
    composed_trf = gryds.ComposedTransformation(bspline_trf, affine_trf)

    z_ind = np.random.randint(image.shape[2])
    t_ind = np.random.randint(2)
    interpolator = gryds.Interpolator(image[...,z_ind,t_ind], mode = 'reflect')

    interpolator_label = gryds.Interpolator(label[...,z_ind,t_ind], order = 0, mode = 'constant')

    patch = interpolator.transform(composed_trf)
    patch_label = interpolator_label.transform(composed_trf)

    if ia:
        intensity_shift = np.random.rand()*.1 - .05
        contrast_shift = np.random.rand()*0.05 + 0.975
        
        patch += intensity_shift
        patch = np.sign(patch)*np.power(np.abs(patch),contrast_shift)

    blur = np.random.uniform() 
    patch = gaussian_filter(patch,sigma=blur)

    # midx = image.shape[0] // 2
    # midy = image.shape[1] // 2
    if patch.shape[0] > sz[0] and patch.shape[1] > sz[1]:
        all_startx = [0, patch.shape[0]//2 - sz[0]//2, patch.shape[0] - sz[0]]
        all_starty = [0, patch.shape[1]//2 - sz[1]//2, patch.shape[1] - sz[1]]
        xrint = np.random.randint(3)
        yrint = np.random.randint(3)
        midx = all_startx[xrint]
        midy = all_starty[yrint]
    
        patch = patch[midx:midx+sz[0],midy:midy+sz[1]]
        patch_label = patch_label[midx:midx+sz[0],midy:midy+sz[1]]
    else:
        patch = patch[:sz[0],:sz[1]]
        patch_label = patch_label[:sz[0],:sz[1]]
        new_patch = np.zeros((sz[0],sz[1]))
        new_patch_label = np.zeros((sz[0],sz[1]))
        new_patch[:patch.shape[0],:patch.shape[1]] = patch
        new_patch_label[:patch_label.shape[0],:patch_label.shape[1]] = patch_label
        patch, patch_label = new_patch, new_patch_label

    # patch = patch[midx-(sz[0]//2):midx+(sz[0]//2),midy-(sz[1]//2):midy+(sz[1]//2)]
    p5, p95 = np.percentile(patch, (5, 95))
    patch = (patch - p5) / (p95 - p5)
    patch = equalize_adapthist(np.clip(patch, 1e-5, 1),kernel_size=24)[...,np.newaxis]
    patch += np.random.normal(scale=0.025, size=patch.shape)

    # patch = np.clip(patch, 0, 1)[...,np.newaxis]

    # patch_label = patch_label[midx-(sz[0]//2):midx+(sz[0]//2),midy-(sz[1]//2):midy+(sz[1]//2)] 

    return (patch, patch_label)

@tf.function
def tf_do_augmentation2(image, label, sz, in_channels, num_classes, ia=False):

    image, label = tf.py_function(func=do_augmentation2, inp=[image, label, sz, ia], Tout=[tf.float32, tf.int32])
    image.set_shape((sz[0], sz[1],in_channels))
    label.set_shape((sz[0], sz[1]))
    one_hot_label = tf.one_hot(label, num_classes)

    return image, one_hot_label

def valid_crop(image, label, sz):

    midx = image.shape[0] // 2
    midy = image.shape[1] // 2
    z_ind = np.random.randint(image.shape[2])
    t_ind = np.random.randint(2)

    if image.shape[0] > sz[0] and image.shape[1] > sz[1]:
        patch = image[midx-(sz[0]//2):midx+(sz[0]//2),midy-(sz[1]//2):midy+(sz[1]//2),z_ind,t_ind]
        patch_label = label[midx-(sz[0]//2):midx+(sz[0]//2),midy-(sz[1]//2):midy+(sz[1]//2),z_ind,t_ind]

    else:
        patch = image[:sz[0],:sz[1],z_ind,t_ind]
        patch_label = label[:sz[0],:sz[1],z_ind,t_ind]
        new_patch = np.zeros((sz[0],sz[1]))
        new_patch_label = np.zeros((sz[0],sz[1]))
        new_patch[:patch.shape[0],:patch.shape[1]] = patch
        new_patch_label[:patch_label.shape[0],:patch_label.shape[1]] = patch_label
        patch, patch_label = new_patch, new_patch_label

    p5, p95 = np.percentile(patch, (5, 95))
    patch = (patch - p5) / (p95 - p5)
    patch = equalize_adapthist(np.clip(patch, 1e-5, 1),kernel_size=24)[...,np.newaxis]
    # patch = np.clip(patch, 1e-5, 1)[...,np.newaxis]

    # patch_label = label[midx-(sz[0]//2):midx+(sz[0]//2),midy-(sz[1]//2):midy+(sz[1]//2),z_ind,t_ind]

    return (patch, patch_label)

@tf.function
def tf_valid_crop(image, label, sz, in_channels, num_classes):

    image, label = tf.py_function(func=valid_crop, inp=[image, label, sz], Tout=[tf.float32, tf.float32])
    image.set_shape((sz[0], sz[1], in_channels))
    label.set_shape((sz[0], sz[1]))
    one_hot_label = tf.one_hot( tf.cast(label, tf.int32), num_classes)

    return image, one_hot_label

def test_crop(image, sz):

    midx = image.shape[0] // 2
    midy = image.shape[1] // 2
    z_ind = np.random.randint(image.shape[2])
    t_ind = np.random.randint(2)

    return image[midx-(sz[0]//2):midx+(sz[0]//2),midy-(sz[1]//2):midy+(sz[1]//2),z_ind,t_ind][...,np.newaxis]

@tf.function
def tf_test_crop(image, sz, in_channels):

    image = tf.py_function(func=test_crop, inp=[image, sz], Tout=tf.float32)
    image.set_shape((sz[0], sz[1], in_channels))
    return image

def do_domain_augmentation2(image, sz):

    random_grid = np.random.rand(2, 7, 7)
    random_grid -= 0.5
    random_grid /= 10
    # Define a B-spline transformation object
    bspline_trf = gryds.BSplineTransformation(random_grid)

    # rotate between -pi/8 and pi/8
    rot = np.random.rand()*np.pi/4 - np.pi/8
    # scale between 0.9 and 1.1
    scale_x = np.random.rand()*0.2 + 0.9
    scale_y = np.random.rand()*0.2 + 0.9
    # translate between -10% and 10%
    trans_x = np.random.rand()*.2 - .1
    trans_y = np.random.rand()*.2 - .1

    affine_trf = gryds.AffineTransformation(
        ndim=2,
        angles=[rot], # the rotation angle
        scaling=[scale_x, scale_y], # the anisotropic scaling
        translation=[trans_x, trans_y], # translation
        center=[0.5, 0.5] # center of rotation
    )
    composed_trf = gryds.ComposedTransformation(bspline_trf, affine_trf)

    z_ind = np.random.randint(image.shape[2])
    t_ind = np.random.randint(2)
    interpolator = gryds.Interpolator(image[...,z_ind,t_ind], mode = 'reflect')

    patch = interpolator.transform(composed_trf)

    patch += np.random.normal(scale=0.025, size=patch.shape)

    blur = np.random.uniform() 
    patch = gaussian_filter(patch,sigma=blur)

    midx = image.shape[0] // 2
    midy = image.shape[1] // 2
    patch = patch[midx-(sz[0]//2):midx+(sz[0]//2),midy-(sz[1]//2):midy+(sz[1]//2)]
    p5, p95 = np.percentile(patch, (5, 95))
    patch = (patch - p5) / (p95 - p5)
    patch = equalize_adapthist(np.clip(patch, 0, 1))[...,np.newaxis]

    return patch

@tf.function
def tf_do_domain_augmentation2(image, label, domain, sz, in_channels, num_classes, num_domains):

    image, label = tf.py_function(func=do_augmentation2, inp=[image, label, sz], Tout=[tf.float32, tf.int32])
    image.set_shape((sz[0], sz[1], in_channels))
    label.set_shape((sz[0], sz[1]))
    one_hot_label = tf.one_hot(label, num_classes)
    domain.set_shape((num_domains))
    
    return image, one_hot_label, domain

@tf.function
def tf_do_domain_only_augmentation2(image, domain, sz, in_channels, num_classes):

    image, _ = tf.py_function(func=do_augmentation2, inp=[image, image, sz], Tout=[tf.float32, tf.int32])
    image.set_shape((sz[0], sz[1], in_channels))
    # label.set_shape((sz[0], sz[1]))
    # one_hot_label = tf.one_hot(label, num_classes)
    domain.set_shape((num_classes))
    
    return image, domain

@tf.function
def tf_do_domain_valid_crop(image, label, domain, sz, in_channels, num_classes, num_domains):
    
    image, label = tf.py_function(func=valid_crop, inp=[image, label, sz], Tout=[tf.float32,tf.float32])
    image.set_shape((sz[0], sz[1], in_channels))
    label.set_shape((sz[0], sz[1]))
    one_hot_label = tf.one_hot( tf.cast(label, tf.int32), num_classes)
    domain.set_shape((num_domains))

    return image, one_hot_label, domain

@tf.function
def tf_do_domain_only_valid_crop(image, domain, sz, in_channels, num_domains):
    
    image, _ = tf.py_function(func=valid_crop, inp=[image, image, sz], Tout=[tf.float32,tf.float32])
    image.set_shape((sz[0], sz[1], in_channels))
    domain.set_shape((num_domains))

    return image, domain