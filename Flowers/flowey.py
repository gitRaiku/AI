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

def datasets(): 
    train_ds = keras.utils.image_dataset_from_directory(
        directory='/home/arch/.keras/datasets/flower_photos',
        validation_split=0.2,
        subset='training',
        seed=123,
        batch_size=b_siz,
        image_size=(img_h, img_w))

    val_ds = keras.utils.image_dataset_from_directory(
        directory='/home/arch/.keras/datasets/flower_photos',
        validation_split=0.2,
        subset='validation',
        seed=123,
        batch_size=b_siz,
        image_size=(img_h, img_w))

    class_names = train_ds.class_names

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return (train_ds, val_ds)

def train():
    train_ds, val_ds = datasets()
    data_augmentation = keras.Sequential([
      layers.RandomRotation(0.2),
      layers.RandomFlip("horizontal_and_vertical"),
      layers.RandomRotation(0.2),
      layers.RandomZoom(0.1)])

    model = Sequential([
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.1),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.1),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(len(class_names))
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.build((32, img_h, img_w, 3))
    model.summary()

    epochs = 40
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )
    model.save('kms.keras')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

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
    plt.show()
    return model

def get():
    return tf.keras.models.load_model('kms40.keras')

def validate():
    _, val_ds = datasets()
    get().evaluate(val_ds)

# daisy/  dandelion/  roses/  sunflowers/  tulips/
def check(model, path):
    ress = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    img = np.array([tf.keras.utils.img_to_array(tf.keras.utils.load_img(path, target_size=(img_h, img_w)))])
    res = model.predict(img)
    print(f'{path} -> ({ress[np.argmax(res)]}) {res}')

model = get()
check(model, "/home/arch/dais.png")
check(model, "/home/arch/dand.png")
check(model, "/home/arch/ros1.png")
check(model, "/home/arch/ros2.png")
check(model, "/home/arch/sun.png")
check(model, "/home/arch/suns.png")
check(model, "/home/arch/tulips.png")

