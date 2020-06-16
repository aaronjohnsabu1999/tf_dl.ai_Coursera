#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from os import path, getcwd, chdir

path = f"{getcwd()}/../Assets/mnist.npz"

def train_mnist():
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = {}):
            try:
                acc = logs.get('acc') > 0.99
            except:
                acc = logs.get('accuracy') > 0.99
            if acc:
                print('Reached 99% accuracy of training data')
                self.model.stop_training = True

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    x_train = x_train / 255.0
    x_test  = x_test  / 255.0
    callback = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = tf.nn.relu),
        tf.keras.layers.Dense( 10, activation = tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        x_train, y_train, epochs = 10, callbacks = [callback]
    )
    
    try:
        return history.epoch, history.history['acc'][-1]
    except:
        return history.epoch, history.history['accuracy'][-1]

train_mnist()