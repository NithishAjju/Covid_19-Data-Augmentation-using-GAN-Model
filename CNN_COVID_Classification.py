# -*- coding: utf-8 -*-

"""
Created on Sun Dec 13 19:37:59 2020
This is CNN model used for COVID classification
@author: cdnguyen
"""


# -*- coding: utf-8 -*-

"""
Created on Sun Dec 13 19:37:59 2020
This is a CNN model used for COVID classification
@author: cdnguyen
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Path for training and testing data
TrainImage = "CovidDataset"
TestImage = "CovidDataset"

# Check if paths exist
if not os.path.exists(TrainImage):
    raise FileNotFoundError(f"Training dataset directory '{TrainImage}' not found.")
if not os.path.exists(TestImage):
    raise FileNotFoundError(f"Testing dataset directory '{TestImage}' not found.")

# Verify dataset structure
Normalimages = os.listdir(os.path.join(TrainImage, "No_findings"))
Pneumonaimages = os.listdir(os.path.join(TrainImage, "pneumonia"))
COVID19images = os.listdir(os.path.join(TrainImage, "Covid-19"))

print(f"Normal images: {len(Normalimages)}")
print(f"Pneumonia images: {len(Pneumonaimages)}")
print(f"COVID-19 images: {len(COVID19images)}")

NUM_TRAINING_IMAGES = len(Normalimages) + len(Pneumonaimages) + len(COVID19images)
print(f"Total training images: {NUM_TRAINING_IMAGES}")

# Image size and batch size
image_size = 32
BATCH_SIZE = 8
epochs = 10

# Steps per epoch
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    rotation_range=15,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
training_set = train_datagen.flow_from_directory(
    TrainImage,
    target_size=(image_size, image_size),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

testing_set = test_datagen.flow_from_directory(
    TestImage,
    target_size=(image_size, image_size),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Dynamically detect the number of classes
num_classes = len(training_set.class_indices)
print(f"Detected {num_classes} classes: {training_set.class_indices}")

# Define the CNN model
def define_model():
    model = Sequential()
    
    # Convolutional Layer
    model.add(Conv2D(32, (3, 3), input_shape=(image_size, image_size, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening Layer
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))  # Match the number of classes dynamically

    # Compile the model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Initialize and compile the model
model = define_model()

# Train the model
history = model.fit(
    training_set,
    validation_data=testing_set,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=epochs,
    verbose=1
)

# Evaluate the model
Y_pred = model.predict(testing_set)
predicted_classes = np.argmax(Y_pred, axis=1)

true_classes = testing_set.classes
class_labels = list(testing_set.class_indices.keys())

# Classification Report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print('Classification Report')
print(report)

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix')
print(conf_matrix)

# Print class indices
print("Class indices:", training_set.class_indices)







