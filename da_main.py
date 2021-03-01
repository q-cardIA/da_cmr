import numpy as np 
import tensorflow as tf
import tensorflow.keras as tfk
import os
from glob import glob

import logging
tf.get_logger().setLevel(logging.ERROR)

import utils
import augmentation
import losses
import u_net2
import base_cnn

TRAIN_DIR = os.path.join('data','labelled')
UNLABELLED_TRAIN_DIR = os.path.join('data','unlabelled')
VALID_DIR = os.path.join('data','valid_labelled')

IMG_DIM = [192, 192]
NO_CHANNELS = 1
NO_CLASSES = 4
NO_DOMAINS = 3
BATCH_SIZE = 16
NO_TRAIN_IMAGES = len(glob(os.path.join(TRAIN_DIR,'*sa.nii.gz'))) 
NO_VALID_IMAGES = len(glob(os.path.join(VALID_DIR,'*sa.nii.gz'))) 
NO_TRAIN_STEPS = NO_TRAIN_IMAGES // BATCH_SIZE
NO_VALID_STEPS = NO_VALID_IMAGES // BATCH_SIZE
NO_FILTERS = 16
RES_STEPS = 5

NO_EPOCHS = 1000

# Images in the dataset will be x,y,z,2 (we have 2 time points ED and ES) 
# since we want only one time point as a training example, we can sample 
# randomly during augmentation. We can do the same random sampling from 
# the z direction to get a 2D input.
# I will take the central 192 x 192 patch for training

in_dim1 = [IMG_DIM[0], IMG_DIM[1], NO_FILTERS]
in_dim2 = [IMG_DIM[0]//(2**(RES_STEPS-1)), IMG_DIM[1]//(2**(RES_STEPS-1)), NO_FILTERS*(2**RES_STEPS)]

cnn_model = base_cnn.define_dual_cnn(in_dim1, in_dim2, NO_DOMAINS)
unet_model = u_net2.UNet2D(in_channels=NO_CHANNELS, out_classes=NO_CLASSES, img_shape = [IMG_DIM[0], IMG_DIM[1], NO_CHANNELS], no_filters=NO_FILTERS, resolution_steps=RES_STEPS)

learning_rate_function_seg = tfk.optimizers.schedules.PiecewiseConstantDecay(boundaries = [150*NO_TRAIN_STEPS, 1000*NO_TRAIN_STEPS], values = [1e-3, 1e-4, 1e-5])
learning_rate_function_class = tfk.optimizers.schedules.PiecewiseConstantDecay(boundaries = [150*NO_TRAIN_STEPS, 400*NO_TRAIN_STEPS], values = [1e-3, 1e-4, 1e-5])

optimizer_seg = tfk.optimizers.Adam(learning_rate_function_seg)
optimizer_class = tfk.optimizers.Adam(learning_rate_function_class)

dice_loss = losses.MultiClassDice()
dice_plus_xent = losses.MultiClassDiceXent()
cat_xent = tfk.losses.CategoricalCrossentropy(from_logits=True)
cat_acc = tf.keras.metrics.CategoricalAccuracy()

domain_dataset = tf.data.Dataset.from_generator(lambda: utils.get_domain_data(TRAIN_DIR,NO_DOMAINS), (tf.float32, tf.float32, tf.float32), \
                                            (tf.TensorShape([None,None,None,None]),tf.TensorShape([None,None,None,None]),tf.TensorShape([None]))).repeat()
domain_dataset = domain_dataset.map(lambda x, y, z: augmentation.tf_do_domain_augmentation2(x, y, z, IMG_DIM, NO_CHANNELS, NO_CLASSES, NO_DOMAINS), 
                                            num_parallel_calls=4).batch(BATCH_SIZE,drop_remainder=True).prefetch(1)
unlabelled_dataset = tf.data.Dataset.from_generator(lambda: utils.get_domain_data_only(UNLABELLED_TRAIN_DIR,NO_DOMAINS), (tf.float32, tf.float32), \
                                        (tf.TensorShape([None,None,None,None]), tf.TensorShape([None]))).repeat()
unlabelled_dataset =  unlabelled_dataset.map(lambda x, y: augmentation.tf_do_domain_only_augmentation2(x, y, IMG_DIM, NO_CHANNELS, NO_DOMAINS), 
                                        num_parallel_calls=4).batch(BATCH_SIZE//4,drop_remainder=True).prefetch(1)
domain_valid_dataset = tf.data.Dataset.from_generator(lambda: utils.get_domain_data(VALID_DIR, NO_DOMAINS), (tf.float32, tf.float32, tf.float32), \
                                            (tf.TensorShape([None,None,None,None]),tf.TensorShape([None,None,None,None]),tf.TensorShape([None]))).repeat()
domain_valid_dataset = domain_valid_dataset.map(lambda x, y, z: augmentation.tf_do_domain_valid_crop(x, y, z, IMG_DIM, NO_CHANNELS, NO_CLASSES, NO_DOMAINS), 
                                        num_parallel_calls=4).batch(BATCH_SIZE,drop_remainder=True).prefetch(1)

alpha = 0.
reverse_alpha = tf.constant([-alpha])

@tf.function
def train_step_seg(inputs, targets):
    
    with tf.GradientTape() as tape:
        y_ = unet_model(inputs, training=True)
        loss_value = dice_plus_xent(y_true=targets, y_pred=y_)
    
    grads = tape.gradient(loss_value, unet_model.trainable_variables)
    optimizer_seg.apply_gradients(zip(grads, unet_model.trainable_variables))

    return loss_value

@tf.function
def train_step_class(inputs, targets):
    
    with tf.GradientTape() as tape:
        y_ = cnn_model(inputs, training=True)
        loss_value = cat_xent(y_true=targets, y_pred=y_)
    
    grads = tape.gradient(loss_value, cnn_model.trainable_variables)
    optimizer_class.apply_gradients(zip(grads, cnn_model.trainable_variables))

    return loss_value

@tf.function
def train_step_adv(inputs, targets):
    with tf.GradientTape() as tape:
        x,y = unet_model.encode(inputs,training=True)
        rep = unet_model.decode(x,y,training=True)
        classified_rep = cnn_model([rep, x] ,training=True)
        loss_value = cat_xent(y_true=targets, y_pred=classified_rep)

    grads_adv = tape.gradient(loss_value, unet_model.trainable_variables)
    reversed_grads_adv = [(reverse_alpha*g, v) if ('conv' in v.name) and ('encode' in v.name or 'decode' in v.name) else (g, v) for g, v in zip(grads_adv, unet_model.trainable_variables)]
    optimizer_seg.apply_gradients(reversed_grads_adv)
    
    return loss_value

iter_data = iter(domain_dataset)
iter_valid_data = iter(domain_valid_dataset)
iter_unlabel_data = iter(unlabelled_dataset)
for epoch in range(NO_EPOCHS):
    epoch_loss_avg_seg = tf.keras.metrics.Mean()
    epoch_loss_avg_class = tf.keras.metrics.Mean()
    epoch_dice = tf.keras.metrics.Mean()
    epoch_class_accuracy = tf.keras.metrics.Mean()
    
    if epoch > 300 and alpha < 1.:
        alpha += 0.0065
        reverse_alpha = tf.constant([-alpha])

    for _ in range(NO_TRAIN_STEPS):
        xs, ys, ds = next(iter_data)
        if epoch < 151 or epoch > 300:
            loss_value_seg = train_step_seg(xs, ys)
            epoch_loss_avg_seg.update_state(loss_value_seg)

        if epoch > 150:
            xu, du = next(iter_unlabel_data)

            x_adv = tf.concat([xs, xu], 0)
            d_adv = tf.concat([ds, du], 0)
            inds = tf.range( BATCH_SIZE + BATCH_SIZE // 4 )
            inds = tf.random.shuffle(inds)
            x_adv = tf.gather(x_adv, inds, axis = 0)
            d_adv = tf.gather(d_adv, inds, axis = 0)

            zclass, feats = unet_model.encode(x_adv)
            xclass = unet_model.decode(zclass, feats)
            loss_value_class = train_step_class([xclass, zclass], d_adv)
            epoch_loss_avg_class.update_state(loss_value_class)

            if alpha > 0:
               loss_value_adv = train_step_adv(x_adv, d_adv)

        y_ = unet_model(xs, training=False)
        epoch_dice.update_state(1 - dice_loss(y_true=ys, y_pred=y_))
        if epoch > 150:
            y_ = cnn_model([xclass, zclass], training=False)
            epoch_class_accuracy.update_state(cat_acc(y_true=d_adv, y_pred=y_))
        else:
            epoch_class_accuracy.update_state(0.0)

    epoch_valid_dice = tf.keras.metrics.Mean()
    for _ in range(NO_VALID_STEPS):
        xv, yv, _ = next(iter_valid_data) 
        y_ = unet_model(xv, training=False)
        epoch_valid_dice.update_state(1 - dice_loss(y_true=yv, y_pred=y_))

    if epoch % 50 == 0:
        unet_model.reset_metrics()
        unet_model.save_weights(os.path.join("da_seg_model_weights", "cp-{:03d}.ckpt".format(epoch)), save_format = 'tf')
        cnn_model.reset_metrics()
        cnn_model.save_weights(os.path.join("da_class_model_weights", "cp-{:03d}.ckpt".format(epoch)), save_format = 'tf')

    print("Epoch {:03d}: Seg Loss: {:.3f}".format(epoch, epoch_loss_avg_seg.result()),flush=True)
    print("Epoch {:03d}: Class Loss: {:.3f}".format(epoch, epoch_loss_avg_class.result()),flush=True)
    print("Epoch {:03d}: Dice: {:.3f}".format(epoch, epoch_dice.result()),flush=True)
    print("Epoch {:03d}: Class Accuracy: {:.3f}".format(epoch, epoch_class_accuracy.result()),flush=True)
    print("Epoch {:03d}: Validation Dice: {:.3f}".format(epoch, epoch_valid_dice.result()),flush=True)

unet_model.reset_metrics()
unet_model.save_weights(os.path.join("da_seg_model_weights", "cp-final.ckpt"), save_format = 'tf')
cnn_model.reset_metrics()
cnn_model.save_weights(os.path.join("da_class_model_weights", "cp-final.ckpt"), save_format = 'tf')
