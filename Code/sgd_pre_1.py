import os
import shutil

# Define the paths
src_folder = 'dtd/image/'
test_folder = 'dtd/image/test/'
train_folder = 'dtd/image/train/'
val_folder = 'dtd/image/val/'

# Create the test and train folders if they don't exist
os.makedirs(test_folder, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Read the filenames from the train and test files
train_filenames = []
test_filenames = []
val_filenames = []

with open('dtd/label/train.txt', 'r') as f:
    train_filenames = f.read().splitlines()

with open('dtd/label/test.txt', 'r') as f:
    test_filenames = f.read().splitlines()
    
with open('dtd/label/val.txt', 'r') as f:
    val_filenames = f.read().splitlines()
    
# Copy the remaining images to the test folder
for filename in test_filenames:
    src_path = os.path.join(src_folder, filename)
    dst_path = os.path.join(test_folder, filename)
    
    # Create the destination directory if it doesn't exist
    destination_dir = os.path.dirname(dst_path)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        
    shutil.copy(src_path, dst_path)
    
# Copy the first num_test images to the test folder
for filename in train_filenames:
    src_path = os.path.join(src_folder, filename)
    dst_path = os.path.join(train_folder, filename)

    # Create the destination directory if it doesn't exist
    destination_dir = os.path.dirname(dst_path)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    shutil.copy(src_path, dst_path)

# Copy the remaining images to the train folder
for filename in val_filenames:
    src_path = os.path.join(src_folder, filename)
    dst_path = os.path.join(val_folder, filename)
    
    # Create the destination directory if it doesn't exist
    destination_dir = os.path.dirname(dst_path)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        
    shutil.copy(src_path, dst_path)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf.config.run_functions_eagerly(True)  # Disable AutoGraph

# Define data directories
train_dir = 'dtd/image/train/'
val_dir = 'dtd/image/val/'
test_dir = 'dtd/image/test/'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten, MaxPool2D, Activation, Dropout

model =  Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=[tf.keras.metrics.Precision()])
model.summary()

# Train model
history = model.fit(train_generator, epochs= 20, verbose = 1, validation_data=val_generator)

# Evaluate model
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print('Test loss: ', test_loss)
print('Test precision:', test_acc)  

def plot_accuracy_loss(history):
    fig = plt.figure(figsize=(15,6))
    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['precision'],'bo--', label = "precision")
    plt.plot(history.history['val_precision'], 'ro--', label = "val_precision")
    plt.title("train_precision vs val_precision")
    plt.ylabel("precision")
    plt.xlabel("epochs")
    plt.legend()

     # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

plot_accuracy_loss(history)