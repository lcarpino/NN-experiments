import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

def alexnet_model(img_shape=(224, 224, 3), nb_classes=1000, weights=None):

    alexnet = Sequential()

    # layer 1
    alexnet.add(Conv2D(filters=96, kernel_size=11, strides=4, padding='same', input_shape=img_shape))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=3, strides=2))

    #layer 2
    alexnet.add(Conv2D(filters=256, kernel_size=5, strides=1, padding="same"))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=3, strides=2))

    # layer 3
    alexnet.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # layer 4
    alexnet.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # layer 5
    alexnet.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=3, strides=2))

    # layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # layer 8
    alexnet.add(Dense(nb_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    if weights is not None:
        lenet.load_weights(weights)

    return alexnet


def mini_alexnet_model(img_shape=(28, 28, 1), nb_classes=10, weights=None):

    alexnet = Sequential()

    # layer 1
    alexnet.add(Conv2D(filters=96, kernel_size=11, strides=4, padding='same', input_shape=img_shape))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=2, strides=1))

    #layer 2
    alexnet.add(Conv2D(filters=256, kernel_size=5, strides=1, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=2, strides=1))

    # layer 3
    alexnet.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # layer 4
    alexnet.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # layer 5
    alexnet.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=2, strides=2))

    # layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(1024))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # layer 7
    alexnet.add(Dense(1024))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # layer 8
    alexnet.add(Dense(nb_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    if weights is not None:
        lenet.load_weights(weights)

    return alexnet
