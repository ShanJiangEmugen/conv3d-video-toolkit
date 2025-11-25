import tensorflow as tf
from tensorflow.keras import Model
from keras.regularizers import l2
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding3D
from tensorflow.keras.optimizers import SGD


import numpy as np
import pandas as pd
import cv2
import glob
import os
import gc
import random


###############################################################################
# Notes:                                                                      #
# this script contain the network of two-stream model.                        #
# There are 2 streams, Spatial Stream which only insert 1 frame each time.    #
# This stream will perform VGG16 ConvNet with ImageNet weights.               #
#                                                                             #
# The other stream is Temporal Stream which take optic flow as input.         #
# This stream will be a time distributed VGG16 for extract features from      #
# motions                                                                     #
#                                                                             #
# At the end, two streams will be fused by algorithms, such as average or more#
# complicated methods.                                                        #
#                                                                             #
# The dataset has extremely imbalanced data, 4% : 96%, can't make correct     #
# prediction without any sampling!!!                                          #
#                                                                             #
#                                                                             #
#                                                                             #
#                                                                             #
# Log:                                                                        #
# Version 0.0: May10, 2022 script created                                     #
# Version 0.1: Mar3, 2022 script was modified and used for tracking mouse     #
#              seizures                                                       #
#                                                                             #
###############################################################################

def c3d_model(n, w, l, c, classes):
    '''
    Using Conv3D instead of TimeDistributed to avoid shape issues
    '''
    input_shape = (n, w, l, c)
    inputs = Input(input_shape)
    weight_decay = 0.001
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_regularizer=l2(weight_decay))(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = MaxPool3D((1, 2, 2), strides=(1, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)           
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = ZeroPadding3D(padding=(0, 1, 1))(x)           
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(4096, kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(4096, kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(classes, kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)

    return model


'''
Following code is copied from 
https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
used for modify layers in keras model
Will use it for transfer learning
'''
def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
    from keras.models import Model

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = Model(layers[0].input, x)
    return new_model

def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    from keras.models import Model

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = Model(layers[0].input, x)
    return new_model



if __name__ == "__main__":
    spatial_stream = c3d_model(11, 15, 360, 240, 3)
    spatial_stream.summary()
    
