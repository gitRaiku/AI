import numpy as np

import tensorflow as t
import tensorflow.keras as keras

rng = t.random.Generator.from_seed(123, alg='philox')

class Augment():
  def __init__(self, seed=42):
    this.seed = seed
  def call(self, inputs, labels):
    inputs = inputs / 255.0
    inputs = t.image.stateless_random_flip_left_right(inputs, seed)
    inputs = t.image.stateless_random_hue(inputs, 0.1, seed)
    inputs = t.image.stateless_random_contrast(inputs, 0.0, 0.2, seed)
    inputs = t.image.stateless_random_brightness(inputs, 0.1, seed)
    labels -= 1
    labels = t.image.stateless_random_flip_left_right(labels, seed)
    return inputs, labels

train_img = keras.utils.image_dataset_from_directory(directory='/home/raiku/.keras/datasets/OxfordPet/images',validation_split=0.2, subset='training', seed=123, batch_size=b_siz, image_size=imgs)

def datasets():
    b_siz = 32
    imgs = (128, 128)

    train_img = keras.utils.image_dataset_from_directory(
        directory='/home/raiku/.keras/datasets/OxfordPet/images',
        validation_split=0.2, 
        subset='training', seed=123, batch_size=b_siz, image_size=imgs)

    train_label = keras.utils.image_dataset_from_directory(
        directory='/home/raiku/.keras/datasets/OxfordPet/annotations',
        validation_split=0.2, 
        subset='training', seed=123, batch_size=b_siz, image_size=imgs)

    val_img = keras.utils.image_dataset_from_directory(
        directory='/home/raiku/.keras/datasets/OxfordPet/images',
        validation_split=0.2, 
        subset='validation', seed=123, batch_size=b_siz, image_size=imgs)

    val_label = keras.utils.image_dataset_from_directory(
        directory='/home/raiku/.keras/datasets/OxfordPet/annotations',
        validation_split=0.2, 
        subset='validation', seed=123, batch_size=b_siz, image_size=imgs)




  image = t.image.resize(datapoint['image'], (128, 128))
  mask = t.image.resize(datapoint['segmentation_mask'], (128, 128), method = t.image.ResizeMethod.NEAREST_NEIGHBOR)

  image, mask = normalize(image, mask)

  return image, mask
