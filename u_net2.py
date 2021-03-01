import numpy as np
import datetime
import tensorflow as tf
import tensorflow.keras as tfk

# ---------------------------- #

class DownBlock(tfk.Sequential):
    
    def __init__(self, j, out_size, kernel_size=3, conv_stride=1, padding='SAME', conv_dilation=1, kernel_initializer='he_normal', 
                bias=False, batch_norm=True):

        super().__init__()

        for i in range(2):    

            self.add(tfk.layers.Conv2D((i+1)*out_size, kernel_size, strides=conv_stride, padding=padding, dilation_rate=conv_dilation, 
                    kernel_initializer=kernel_initializer, use_bias=bias, name="down_conv_{}_{}".format(j + 1, i + 1)))
            if batch_norm:
                self.add(tfk.layers.BatchNormalization(name="down_bn_{}_{}".format(j + 1, i + 1)))
            self.add(tfk.layers.ReLU())

# ---------------------------- #

class UpBlock(tfk.Sequential):
    
    def __init__(self, j, out_size, kernel_size=3, conv_stride=1, padding='SAME', conv_dilation=1, kernel_initializer='he_normal', 
                bias=False, batch_norm=True):

        super().__init__()

        for i in range(2):    

            self.add(tfk.layers.Conv2D(out_size, kernel_size, strides=conv_stride, padding=padding, dilation_rate=conv_dilation, 
                    kernel_initializer=kernel_initializer, use_bias=bias, name="up_conv_{}_{}".format(j + 1, i + 1)))
            if batch_norm:
                self.add(tfk.layers.BatchNormalization(name="up_bn_{}_{}".format(j + 1, i + 1)))
            self.add(tfk.layers.ReLU())

# ---------------------------- #

class Encode(tfk.Sequential):

    def __init__(self, filters, resolution_steps = 4, kernel_size = 3, conv_stride=1, padding='SAME', conv_dilation=1, kernel_initializer='he_normal', 
                pool_size=(2, 2), pool_strides=None, bias=False, batch_norm=True):

        super().__init__()
        self.resolution_steps = resolution_steps
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        
        for i in range(self.resolution_steps):
            self.add( DownBlock(i, filters, kernel_size, conv_stride, padding, conv_dilation, kernel_initializer, bias, batch_norm) )
            filters *= 2
            

    def call(self, x):

        layer_features = []
        for i in range(self.resolution_steps):
            
            x = self.get_layer(index = i)(x)

            if i < self.resolution_steps - 1:
                layer_features.append(x)
                x = tfk.layers.MaxPool2D(pool_size=self.pool_size, strides=self.pool_strides)(x)

        return x, layer_features

# ---------------------------- #

class UpScale(tfk.Sequential):

    def __init__(self, j, out_size, padding='SAME', kernel_initializer='he_normal', 
                bias=False, tconv_kernel_size=2, tconv_stride=2):
        
        super().__init__()
        self.tconv_kernel_size = tconv_kernel_size
        self.tconv_stride = tconv_stride
        self.bias = bias
        self.add(tfk.layers.Conv2DTranspose(out_size, tconv_kernel_size, tconv_stride, padding=padding, kernel_initializer=kernel_initializer, use_bias=bias, name="up_tconv_{}".format(j + 1)))

    def call(self, x, xskip):

        x = self.get_layer(index = 0)(x)
        x = tf.concat([xskip, x],axis=-1)

        return x

# ---------------------------- #

class Decode(tfk.Sequential):

    def __init__(self, filters, resolution_steps = 4, kernel_size = 3, conv_stride = 1, padding='SAME', conv_dilation=1, kernel_initializer='he_normal', 
                 bias=False, batch_norm=True, tconv_kernel_size=2, tconv_stride=2):

        super().__init__()
        self.resolution_steps = resolution_steps
        
        for i in range(resolution_steps-1):
            
            self.add( UpScale(i, int(filters), padding, kernel_initializer, bias, tconv_kernel_size, tconv_stride  ) )
            filters /= 2
            self.add( UpBlock(i, int(filters), kernel_size, conv_stride, padding, conv_dilation, kernel_initializer, bias, batch_norm) )


    def call(self, x, layer_features):

        for i in range(self.resolution_steps - 1):

            xskip = layer_features[-(i+1)]
            x = self.get_layer(index = 2*i)(x, xskip)
            x = self.get_layer(index = 2*i + 1)(x)

        return x

# ---------------------------- # 

class UNet2D(tfk.Model):

    def __init__(self, in_channels, out_classes, img_shape, no_filters, resolution_steps = 4, convs_per_resolution = 2, kernel_size = 3, conv_stride=1, 
                conv_dilation = 1, padding='SAME', pool_size=(2, 2), pool_strides=None, batch_norm=True, tconv_kernel_size=2, 
                tconv_stride=2, kernel_initializer='he_normal', bias=False):

        super().__init__()
        self.resolution_steps = resolution_steps
        self.convs_per_resolution = convs_per_resolution
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.img_shape = img_shape


        self.encode = Encode(no_filters, resolution_steps, kernel_size, conv_stride, padding, conv_dilation, kernel_initializer, pool_size, pool_strides, bias, batch_norm)
        self.decode = Decode(no_filters*(2**(resolution_steps-1)), resolution_steps, kernel_size, conv_stride, padding, conv_dilation, kernel_initializer, bias, batch_norm, tconv_kernel_size, tconv_stride) 
        self.output_layer = tfk.layers.Conv2D(out_classes, kernel_size=1, kernel_initializer=kernel_initializer, use_bias=bias)


    def call(self, x):
        
        # x = self.encode(x)
        x, feats = self.encode(x)
        x = self.decode(x, feats)
        x = self.output_layer(x)

        return x

    