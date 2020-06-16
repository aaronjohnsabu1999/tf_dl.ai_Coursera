#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = f"{getcwd()}/../Assets/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("../Assets/h-or-s")
zip_ref.close()

def train_happy_sad_model():
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = {}):
            try:
                acc = logs.get('acc') > DESIRED_ACCURACY
            except:
                acc = logs.get('accuracy') > DESIRED_ACCURACY
            if acc:
                print('Reached ' + str(DESIRED_ACCURACY*100) + '% accuracy so cancelling training!')
                self.model.stop_training = True
        
    callbacks = myCallback()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dense(  1, activation = 'sigmoid')
    ])
    model.compile(optimizer = RMSprop(lr = 0.001),
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'])
    
    train_datagen = ImageDataGenerator(rescale = 1.0/255)
    train_generator = train_datagen.flow_from_directory(
        '../Assets/h-or-s/',
        target_size = (150,150),
        batch_size = 10,
        class_mode = 'binary'
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 8,
        epochs = 20,
        callbacks = [callbacks],
        verbose = 1
    )

    try:
        return history.history['acc'][-1]
    except:
        return history.history['accuracy'][-1]

train_happy_sad_model()