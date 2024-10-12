# backend/train_model.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Define constants
IMG_HEIGHT = 416
IMG_WIDTH = 416
BATCH_SIZE = 32
EPOCHS = 10

# Load the dataset
train_data = pd.read_csv('dataset/train/_classes.csv')
train_data.columns = train_data.columns.str.strip()  # Strip whitespace from headers
train_images_dir = 'dataset/train/'

# Create a mapping from filenames to labels
def get_labels(data):
    labels = []
    for index, row in data.iterrows():
        if row['Fresh'] == 1:
            labels.append(0)  # Fresh
        elif row['Half-Fresh'] == 1:
            labels.append(1)  # Half-Fresh
        else:
            labels.append(2)  # Spoiled
    return labels

train_labels = get_labels(train_data)

# Load and preprocess images
def load_images(data_dir, filenames):
    images = []
    for filename in filenames:
        img_path = os.path.join(data_dir, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
    return np.array(images)

train_images = load_images(train_images_dir, train_data['filename'])

# Normalize images
train_images = train_images / 255.0  # Scale pixel values to [0, 1]

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Output layer with 3 classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, np.array(train_labels), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save the model
model.save('model.h5')
