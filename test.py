import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import numpy as np
from model.lenet import lenet_model

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# normalise the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

lenet = lenet_model()

lenet.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

lenet.fit(train_images, train_labels, validation_split=0.2, epochs=50, batch_size=256)
test_loss, test_acc = lenet.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
