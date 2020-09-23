# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:53:26 2020

@author: evanj
"""
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


#test restoring model from file
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2),strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10)])

model.load_weights(filepath='./checkpoints/my_checkpoint')
model.summary()