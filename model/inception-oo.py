import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, InputLayer, Dense, Dropout, Activation, Flatten, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization

class InceptionBlock(Layer):

    def __init__(self, filters_1x1, filters_3x3_reduce, filters_3x3,
                       filters_5x5_reduce, filters_5x5, pool_proj, **kwargs):
        super(InceptionBlock, self).__init__(**kwargs)

        self.filters_1x1 = filters_1x1
        self.filters_3x3_reduce = filters_3x3_reduce
        self.filters_3x3 = filters_3x3
        self.filters_5x5_reduce = filters_5x5_reduce
        self.filters_5x5 = filters_5x5
        self.pool_proj = pool_proj

        self.conv_1x1 = Conv2D(filters_1x1, kernel_size=1, padding='same',
                               activation='relu')

        self.dred_3x3 = Conv2D(filters_3x3_reduce, kernel_size=1,
                               padding='same', activation='relu')
        self.conv_3x3 = Conv2D(filters_3x3, kernel_size=3, padding='same',
                               activation='relu')

        self.dred_5x5 = Conv2D(filters_5x5_reduce, kernel_size=1,
                               padding='same', activation='relu')
        self.conv_5x5 = Conv2D(filters_5x5, kernel_size=5, padding='same',
                               activation='relu')

        self.pool_3x3 = MaxPooling2D(pool_size=3, strides=1, padding='same')
        self.pred_3x3 = Conv2D(pool_proj, kernel_size=1, padding='same',
                               activation='relu')

    def call(self, inputs):

        stack_1 = self.conv_1x1(inputs)

        stack_2 = self.dred_3x3(inputs)
        stack_2 = self.conv_3x3(stack_2)

        stack_3 = self.dred_5x5(inputs)
        stack_3 = self.conv_5x5(stack_3)

        stack_4 = self.pool_3x3(inputs)
        stack_4 = self.pred_3x3(stack_4)

        output = concatenate([stack_1, stack_2, stack_3, stack_4])

        return output
