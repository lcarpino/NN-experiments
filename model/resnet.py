import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Add, Dense
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization

def bottleneck(layer_in, filters, stride=1, short='identity'):

    if short == 'identity':
        shortcut = layer_in

    elif short == 'conv':
        shortcut = Conv2D(4 * filters, kernel_size=1, strides=stride)(layer_in)
        shortcut = BatchNormalization()(shortcut)

    layer_out = Conv2D(filters, kernel_size=1, strides=stride)(layer_in)
    layer_out = BatchNormalization()(layer_out)
    layer_out = Activation('relu')(layer_out)

    layer_out = Conv2D(filters, kernel_size=3, padding='same')(layer_out)
    layer_out = BatchNormalization()(layer_out)
    layer_out = Activation('relu')(layer_out)

    layer_out = Conv2D(4*filters, kernel_size=1)(layer_out)
    layer_out = BatchNormalization()(layer_out)
    layer_out = Activation('relu')(layer_out)

    layer_out = Add()([shortcut, layer_out])
    layer_out = Activation('relu')(layer_out)

    return layer_out

def stacked_bottleneck(layer_in, filters, blocks, stride_1=2):

    layer_out = bottleneck(layer_in, filters, stride=stride_1, short='conv')

    for i in range(2, blocks+1):
        layer_out = bottleneck(layer_out, filters, short='identity')

    return layer_out

def resnet_stem(layer_in):

    layer_out = ZeroPadding2D(padding=3)(layer_in)
    layer_out = Conv2D(64, kernel_size=7, strides=2)(layer_out)
    layer_out = BatchNormalization()(layer_out)
    layer_out = Activation('relu')(layer_out)
    layer_out = ZeroPadding2D(padding=1)(layer_out)
    layer_out = MaxPooling2D(pool_size=3, strides=2)(layer_out)

    return layer_out

def resnet_classifier(layer_in, nb_classes):

    layer_out = GlobalAveragePooling2D()(layer_in)
    output = Dense(nb_classes, activation='softmax')(layer_out)

    return output

def resnet50_model(img_shape=(224, 224, 3), nb_classes=1000, weights=None):

    input = Input(shape=img_shape)

    layer_out = resnet_stem(input)
    layer_out = stacked_bottleneck(layer_out, filters=64, blocks=3, stride_1=1)
    layer_out = stacked_bottleneck(layer_out, filters=128, blocks=4, stride_1=2)
    layer_out = stacked_bottleneck(layer_out, filters=256, blocks=6, stride_1=2)
    layer_out = stacked_bottleneck(layer_out, filters=512, blocks=3, stride_1=2)

    output = resnet_classifier(layer_out, nb_classes)

    resnet = Model(inputs=input, outputs=output, name='resnet-50')

    return resnet
