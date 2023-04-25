"""
Contains code for the NetSeg neural network architecture.
"""
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *

class unet_4(tf.keras.Model):
    """
    Class for the u-net model I use for segmentation.
    """
    
    def __init__(self, input_shape = (None,96,112,96), num_classes = 7):
        """
        Initialize u-net model.

        Args:
            input_shape: tuple describing input shape of 3D images.
            num_classes: int, number of segmentation classes

        Returns:
            An initialized u-net model.
        """
        
        super(unet_4, self).__init__()

        self.num_classes=num_classes

        # ENCODER

        self.input1 = InputLayer(input_shape = input_shape)
        self.conv1_0 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.conv1_1 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.pool1_0 = MaxPooling3D(pool_size=(2,2,2))

        self.conv2_0 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.conv2_1 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.pool2_0 = MaxPooling3D(pool_size=(2,2,2))
        
        self.conv3_0 = Conv3D(128, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.conv3_1 = Conv3D(128, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.pool3_0 = MaxPooling3D(pool_size=(2,2,2))

        # BOTTOM

        self.conv4_0 = Conv3D(256, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.conv4_1 = Conv3D(256, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.drop4_0 = Dropout(0.5)
    
        # DECODER
   
        self.up5_0 = Conv3DTranspose(128, 2, strides = 2, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer= 'GlorotNormal')
        self.conv5_0 = Conv3D(128, 2, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.conv5_1 = Conv3D(128, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
   
        self.up6_0 = Conv3DTranspose(64, 2, strides = 2, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer= 'GlorotNormal')
        self.conv6_0 = Conv3D(64, 2, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.conv6_1 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
  
        self.up7_0 = Conv3DTranspose(32, 2, strides = 2, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer= 'GlorotNormal')
        self.conv7_0 = Conv3D(32, 2, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.conv7_1 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
        self.conv7_2 = Conv3D(num_classes, 3, activation = 'softmax', padding = 'same', data_format='channels_last',  kernel_initializer = 'GlorotNormal')
    
    def call(self, inputs, training_bool = True):
        """
        Passes an input "forward" through the model and produces an inference.

        Args:
            inputs: 3D tensor comprising the input image
            training_bool: whether or not the model is training during this forward pass.
                If true, affects dropout layers. 
        
        Returns:
            Output segmentation probability map.
        """
        level1 = self.input1(inputs)
        level1 = self.conv1_0(level1)
        level1 = self.conv1_1(level1)

        level2 = self.pool1_0(level1)
        level2 = self.conv2_0(level2)
        level2 = self.conv2_1(level2)

        level3 = self.pool2_0(level2)
        level3 = self.conv3_0(level3)
        level3 = self.conv3_1(level3)

        bottom = self.pool3_0(level3)
        bottom = self.conv4_0(bottom)
        bottom = self.conv4_1(bottom)
        bottom = self.drop4_0(inputs=bottom, training = training_bool)

        bottom = self.up5_0(bottom)
        
        level3 = tf.concat([level3, bottom], axis = 4)
        level3 = self.conv5_0(level3)
        level3 = self.conv5_1(level3)

        level3 = self.up6_0(level3)
        
        level2 = tf.concat([level2, level3], axis=4)
        level2 = self.conv6_0(level2)
        level2 = self.conv6_1(level2)

        level2 = self.up7_0(level2)

        level1 = tf.concat([level1, level2], axis = 4)
        level1 = self.conv7_0(level1)
        level1 = self.conv7_1(level1)
        level1 = self.conv7_2(level1)
        
        return level1

