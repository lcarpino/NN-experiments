import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization


def lenet1_model(img_shape=(28, 28, 1), nb_classes=10, weights=None):

    lenet = Sequential()

    lenet.add(Conv2D(filters=4,  kernel_size=(5, 5), strides=1, activation='tanh', input_shape=img_shape))
    lenet.add(AveragePooling2D(pool_size=(2, 2)))

    lenet.add(Conv2D(filters=12, kernel_size=(5, 5), strides=1, activation='tanh'))
    lenet.add(AveragePooling2D(pool_size=(2, 2)))

    lenet.add(Conv2D(filters=nb_classes, kernel_size=(4, 4), activation='softmax'))
    lenet.add(Flatten())

    if weights is not None:
        lenet.load_weights(weights)

    return lenet


def lenet4_model(img_shape=(28, 28, 1), nb_classes=10, weights=None):

    lenet = Sequential()

    lenet.add(Conv2D(filters=4,  kernel_size=5, strides=1, activation='tanh', padding='same', input_shape=img_shape))
    lenet.add(AveragePooling2D(pool_size=2, strides=2))

    lenet.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh', padding='valid'))
    lenet.add(AveragePooling2D(pool_size=2, strides=2))

    lenet.add(Flatten())
    lenet.add(Dense(120, activation='tanh'))
    lenet.add(Dense(nb_classes, activation='softmax'))

    if weights is not None:
        lenet.load_weights(weights)

    return lenet


def lenet5_model(img_shape=(28, 28, 1), nb_classes=10, weights=None):

    lenet = Sequential()

    lenet.add(Conv2D(filters=6,  kernel_size=5, strides=1, activation='tanh', padding='same', input_shape=img_shape))
    lenet.add(AveragePooling2D(pool_size=2, strides=2))

    lenet.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh', padding='valid'))
    lenet.add(AveragePooling2D(pool_size=2, strides=2))

    lenet.add(Flatten())
    lenet.add(Dense(120, activation='tanh'))

    lenet.add(Dense(84, activation='tanh'))

    lenet.add(Dense(nb_classes, activation='softmax'))

    if weights is not None:
        lenet.load_weights(weights)

    return lenet


def lenet_model(img_shape=(28, 28, 1), nb_classes=10, weights=None):

    lenet = Sequential()

    # layer 1
    lenet.add(Conv2D(filters=6, kernel_size=5, strides=1, padding='same', input_shape=img_shape))
    lenet.add(BatchNormalization())
    lenet.add(Activation('relu'))
    lenet.add(MaxPooling2D(pool_size=2, strides=2))
    lenet.add(Dropout(0.25))

    # layer 2
    lenet.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='valid'))
    lenet.add(BatchNormalization())
    lenet.add(Activation('relu'))
    lenet.add(MaxPooling2D(pool_size=2, strides=2))
    lenet.add(Dropout(0.25))

    # layer 3
    lenet.add(Flatten())
    lenet.add(Dense(120))
    lenet.add(BatchNormalization())
    lenet.add(Activation('relu'))
    lenet.add(Dropout(0.4))

    # layer 4
    lenet.add(Dense(84))
    lenet.add(BatchNormalization())
    lenet.add(Activation('relu'))
    lenet.add(Dropout(0.4))

    # layer 5
    lenet.add(Dense(nb_classes))
    lenet.add(BatchNormalization())
    lenet.add(Activation('softmax'))

    if weights is not None:
        lenet.load_weights(weights)

    return lenet
