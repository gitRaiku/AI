#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as callbacks

# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
# train_dir = os.path.join(PATH, 'train')
# validation_dir = os.path.join(PATH, 'validation')

bsize = 32
imgsize = (160, 160)

def datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        directory='/home/raiku/.keras/datasets/cats_and_dogs_filtered/train',
        seed=123,
        batch_size=bsize,
        image_size=imgsize)

    val_ds = keras.utils.image_dataset_from_directory(
        directory='/home/raiku/.keras/datasets/cats_and_dogs_filtered/validation',
        seed=123,
        batch_size=bsize,
        image_size=imgsize)

    class_names = train_ds.class_names

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return (train_ds, val_ds, class_names)

def train():
    train_ds, val_ds, class_names = datasets()
    val_batches = tf.data.experimental.cardinality(val_ds)
    ttes_ds = val_ds.take(val_batches // 5)
    tval_ds = val_ds.skip(val_batches // 5)
    ttes_ds = ttes_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    tval_ds = tval_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    data_augmentation = keras.Sequential([
      layers.RandomRotation(0.1),
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.2),
      layers.RandomZoom(0.1)])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    imgshape = imgsize + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=imgshape,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    initial_epochs = 10
    history = model.fit(train_ds,
                        epochs=initial_epochs,
                        validation_data=tval_ds)
    model.save('kms_1.keras')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def get():
    return keras.models.load_model('kms_1.keras')

def test_step(model, images, labels):
  predictions = model(images, training=False)
  for i in range(32):
      try:
          r = 1 if predictions[i][0].numpy() >= 0.5 else 0
          l = labels[i].numpy()
          if r != l:
              img = images.numpy()[i]
              for x in range(len(img)):
                  for y in range(len(img[x])):
                      img[x][y] /= 255.0
              kms = ['cat', 'dog']
              print(f'Label {kms[l]}, prediction {kms[r]}({predictions[i][0].numpy()})')
              plt.imshow(img)
              plt.show()
      except Exception:
          # print(e)
          pass

def evl(model):
    _, val_ds, _ = datasets()
    '''
    le = len(val_ds)
    for idx, (val_images, val_labels) in enumerate(val_ds):
      print(f'{idx} / {le}', end='\r')
      
      test_step(model, val_images, val_labels)
    '''
    model.evaluate(val_ds)


def check(model, path):
    ress = ['cat', 'dog']
    img = np.array([tf.keras.utils.img_to_array(tf.keras.utils.load_img(path, target_size=(160, 160)))])
    res = model.predict(img)
    print(f'{path} -> ({ress[1 if (np.argmax(res) > 0.5) else 0]}) {res}')

# evl(get())
model = get()
check(model, '/home/raiku/mpv-shot0003.jpg')


# train()
