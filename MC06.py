import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam

data_dir = pathlib.Path('.')
batch_size = 32
pix=256

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(pix, pix),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(pix, pix),
  batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
model = Sequential([
  layers.Conv2D(16, (7,7), strides=(2,2), padding="valid", input_shape=(pix, pix, 3), kernel_initializer="he_normal", kernel_regularizer=l2(0.0005)),
  layers.Conv2D(32, (3,3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.0005)),
  layers.BatchNormalization(),
  layers.Conv2D(32, (3,3), strides=(2,2), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.0005)),
  layers.BatchNormalization(),
  layers.Dropout(0.25),
  layers.Conv2D(64, (3,3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.0005)),
  layers.BatchNormalization(),
  layers.Conv2D(64, (3,3), strides=(2,2), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.0005)),
  layers.BatchNormalization(),
  layers.Dropout(0.25),
  layers.Conv2D(128, (3,3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.0005)),
  layers.BatchNormalization(),
  layers.Conv2D(128, (3,3), strides=(2,2), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.0005)),
  layers.BatchNormalization(),
  layers.Dropout(0.25),
  layers.Flatten(),
  layers.Dense(512, activation="relu", kernel_initializer="he_normal"),
  layers.BatchNormalization(),
  layers.Dropout(0.5),
  layers.Dense(num_classes)
])

epochs=10
model.compile(optimizer=Adam(lr=1e-4, decay=1e-4 / epochs),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

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
