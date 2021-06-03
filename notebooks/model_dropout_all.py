#import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Lambda, GlobalAveragePooling2D, concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential, load_model

# Build U-Net model

def unet_dropout_all(pretrained_weights = None, input_size = (256,256,3), dropout = 0.2, hn = 'he_normal', reducing_factor = 8, n_classes = 5):

    inputs = Input(input_size)

    conv1 = Conv2D(int(64/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(inputs)
    conv1 = Conv2D(int(64/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv1)
    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    
    conv2 = Conv2D(int(128/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(pool1)
    conv2 = Conv2D(int(128/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv2)
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    
    conv3 = Conv2D(int(256/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(pool2)
    conv3 = Conv2D(int(256/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv3)
    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    
    conv4 = Conv2D(int(512/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(pool3)
    conv4 = Conv2D(int(512/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(int(1024/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(pool4)
    conv5 = Conv2D(int(1024/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(int(512/reducing_factor), 2, activation = 'relu', padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(int(512/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(merge6)
    conv6 = Conv2D(int(512/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv6)
    drop6 = Dropout(dropout)(conv6)

    up7 = Conv2D(int(256/reducing_factor), 2, activation = 'relu', padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(drop6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(int(256/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(merge7)
    conv7 = Conv2D(int(256/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv7)
    drop7 = Dropout(dropout)(conv7)

    up8 = Conv2D(int(128/reducing_factor), 2, activation = 'relu', padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(drop7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(int(128/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(merge8)
    conv8 = Conv2D(int(128/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv8)
    drop8 = Dropout(dropout)(conv8)

    up9 = Conv2D(int(64/reducing_factor), 2, activation = 'relu', padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(drop8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(int(64/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(merge9)
    conv9 = Conv2D(int(64/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv9)
    conv9 = Conv2D(int(64/reducing_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv9)
    drop9 = Dropout(dropout)(conv9)

    conv10 = Conv2D(n_classes, 1, activation = 'softmax')(drop9)

    model = Model(inputs = inputs, outputs = conv10)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model