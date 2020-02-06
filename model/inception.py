import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization


def naive_inception_block(layer_in, filters_1x1, filters_3x3, filters_5x5):

    conv_1x1 = Conv2D(filters_1x1, kernel_size=1, padding='same', activation='relu')(layer_in)
    conv_3x3 = Conv2D(filters_3x3, kernel_size=3, padding='same', activation='relu')(layer_in)
    conv_5x5 = Conv2D(filters_5x5, kernel_size=5, padding='same', activation='relu')(layer_in)
    pool_3x3 = MaxPooling2D(pool_size=3, strides=1, padding='same')(layer_in)

    layer_out = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_3x3], axis=-1)

    return layer_out

def inception_block(layer_in, filters_1x1, filters_3x3_reduce, filters_3x3,
                    filters_5x5_reduce, filters_5x5, pool_proj,
                    kernel_init=keras.initializers.glorot_uniform(),
                    bias_init=keras.initializers.Constant(value=0.2)):

    conv_1x1 = Conv2D(filters_1x1, kernel_size=1, padding='same', activation='relu',
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(layer_in)

    conv_3x3 = Conv2D(filters_3x3_reduce, kernel_size=1, padding='same', activation='relu',
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(layer_in)
    conv_3x3 = Conv2D(filters_3x3, kernel_size=3, padding='same',
                      activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, kernel_size=1, padding='same',
                      activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(layer_in)
    conv_5x5 = Conv2D(filters_5x5, kernel_size=5, padding='same',
                      activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_5x5)

    pool_3x3 = MaxPooling2D(pool_size=3, strides=1, padding='same')(layer_in)
    pool_3x3 = Conv2D(pool_proj, kernel_size=1, padding='same',
                      activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(pool_3x3)

    layer_out = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_3x3], axis=-1)

    return layer_out

def inception_stem(layer_in):

    layer_out = Conv2D(filters=64, kernel_size=7, strides=2, padding='valid', activation='relu')(layer_in)
    layer_out = MaxPooling2D(pool_size=3, strides=2, padding='valid')(layer_out)
    layer_out = Conv2D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu')(layer_out)
    layer_out = Conv2D(filters=192, kernel_size=3, strides=1, padding='valid', activation='relu')(layer_out)
    layer_out = MaxPooling2D(pool_size=3, strides=2, padding='valid')(layer_out)

    return layer_out

def mini_inception_stem(layer_in):

    layer_out = Conv2D(filters=16, kernel_size=3, strides=2, padding='valid', activation='relu')(layer_in)
    layer_out = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu')(layer_out)

    return layer_out

def inception_auxiliary_classifier(layer_in, nb_classes):

    layer_out = AveragePooling2D(pool_size=5, strides=3)(layer_in)
    layer_out = Conv2D(filters=128, kernel_size=1, padding='valid', activation='relu')(layer_out)
    layer_out = Flatten()(layer_out)
    layer_out = Dense(1024, activation='relu')(layer_out)
    layer_out = Dropout(0.7)(layer_out)
    output = Dense(nb_classes, activation='softmax')(layer_out)

    return output

def mini_inception_auxiliary_classifier(layer_in, nb_classes):

    layer_out = AveragePooling2D(pool_size=5, strides=3)(layer_in)
    layer_out = Conv2D(filters=32, kernel_size=1, padding='valid', activation='relu')(layer_out)
    layer_out = Flatten()(layer_out)
    layer_out = Dense(128, activation='relu')(layer_out)
    layer_out = Dropout(0.7)(layer_out)
    output = Dense(nb_classes, activation='softmax')(layer_out)

    return output

def inception_classifier(layer_in, nb_classes):

    layer_out = GlobalAveragePooling2D()(layer_in)
    layer_out = Dropout(0.4)(layer_out)
    layer_out = Flatten()(layer_out)
    output = Dense(nb_classes, activation='softmax')(layer_out)

    return output

def inception_model(img_shape=(224, 224, 3), nb_classes=1000, weights=None):

    input = Input(shape=img_shape)

    layer_out = inception_stem(input)
    layer_out = inception_block(layer_out, 64, 96, 128, 16, 32, 32)                # inception_3a
    layer_out = inception_block(layer_out, 128, 128, 192, 32, 96, 64)              # inception_3b
    layer_out = MaxPooling2D(pool_size=3, strides=2, padding='valid')(layer_out)
    layer_out = inception_block(layer_out, 192, 96, 208, 16, 48, 64)               # inception_4a
    aux0      = inception_auxiliary_classifier(layer_out, nb_classes)
    layer_out = inception_block(layer_out, 160, 112, 224, 24, 64, 64)              # inception_4b
    layer_out = inception_block(layer_out, 128, 128, 256, 24, 64, 64)              # inception_4c
    layer_out = inception_block(layer_out, 112, 144, 288, 32, 64, 64)              # inception_4d
    aux1      = inception_auxiliary_classifier(layer_out, nb_classes)
    layer_out = inception_block(layer_out, 256, 160, 320, 32, 128, 128)            # inception_4e
    layer_out = MaxPooling2D(pool_size=3, strides=2, padding='valid')(layer_out)
    layer_out = inception_block(layer_out, 256, 160, 320, 32, 128, 128)            # inception_5a
    layer_out = inception_block(layer_out, 384, 192, 384, 48, 128, 128)            # inception_5b
    output    = inception_classifier(layer_out, nb_classes)

    inception = Model(inputs=input, outputs=[output, aux0, aux1], name='Inception-v1')

    return inception

def mini_inception_model(img_shape=(28, 28, 1), nb_classes=10, weights=None):

    input = Input(shape=img_shape)

    layer_out = mini_inception_stem(input)
    layer_out = inception_block(layer_out, 64, 96, 128, 16, 32, 32)                # inception_3a
    layer_out = inception_block(layer_out, 128, 128, 192, 32, 96, 64)              # inception_3b
    layer_out = MaxPooling2D(pool_size=3, strides=2, padding='valid')(layer_out)
    layer_out = inception_block(layer_out, 192, 96, 208, 16, 48, 64)               # inception_4a
    aux0      = mini_inception_auxiliary_classifier(layer_out, nb_classes)
    layer_out = inception_block(layer_out, 160, 112, 224, 24, 64, 64)              # inception_4b
    layer_out = inception_block(layer_out, 128, 128, 256, 24, 64, 64)              # inception_4c
    layer_out = inception_block(layer_out, 112, 144, 288, 32, 64, 64)              # inception_4d
    aux1      = mini_inception_auxiliary_classifier(layer_out, nb_classes)
    layer_out = inception_block(layer_out, 256, 160, 320, 32, 128, 128)            # inception_4e
    layer_out = MaxPooling2D(pool_size=3, strides=2, padding='valid')(layer_out)
    layer_out = inception_block(layer_out, 256, 160, 320, 32, 128, 128)            # inception_5a
    layer_out = inception_block(layer_out, 384, 192, 384, 48, 128, 128)            # inception_5b
    output    = inception_classifier(layer_out, nb_classes)

    inception = Model(inputs=input, outputs=[output, aux0, aux1], name='Mini-Inception-v1')

    return inception

