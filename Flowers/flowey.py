#!/bin/python3 

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

print('Finished import')

# https://www.tensorflow.org/tutorials/images/classification
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
# data_dir = pathlib.Path(data_dir).with_suffix('')

img_h = 180
img_w = 180
b_siz = 32

train_ds = keras.utils.image_dataset_from_directory(
    directory='/home/raiku/.keras/datasets/flower_photos',
    validation_split=0.2,
    subset='training',
    seed=123,
    batch_size=b_siz,
    image_size=(img_h, img_w))

val_ds = keras.utils.image_dataset_from_directory(
    directory='/home/raiku/.keras/datasets/flower_photos',
    validation_split=0.2,
    subset='validation',
    seed=123,
    batch_size=b_siz,
    image_size=(img_h, img_w))

class_names = train_ds.class_names

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def train():
    model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(len(class_names))
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=10
    )
    model.save('kms.keras')

def get():
    return tf.keras.models.load_model('kms.keras')

model = get()

'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()'''
