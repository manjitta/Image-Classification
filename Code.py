# -*- coding: utf-8 -*-
"""

"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications import MobileNetV2
from keras.optimizers import Adam
from keras.models import Model
from keras.engine import Input

import os
import shutil
from sklearn.model_selection import train_test_split

main_folder = 'path_to_main_folder'
train_folder = '/FoodImage/train'
validation_folder = '/FoodImage/validation'

# Create train and validation folders if they don't exist
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(validation_folder):
    os.makedirs(validation_folder)

# List all class subfolders in the main folder
class_folders = [folder for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]

# Split data into train and validation sets for each class
for class_folder in class_folders:
    class_path = os.path.join(main_folder, class_folder)
    train_class_path = os.path.join(train_folder, class_folder)
    validation_class_path = os.path.join(validation_folder, class_folder)
    
    # Create train and validation subfolders for the current class
    if not os.path.exists(train_class_path):
        os.makedirs(train_class_path)
    if not os.path.exists(validation_class_path):
        os.makedirs(validation_class_path)
    
    # List images in the current class folder
    class_images = os.listdir(class_path)
    
    # Split data using sklearn's train_test_split
    train_images, validation_images = train_test_split(class_images, test_size=0.3, random_state=42)
    
    # Move images to train and validation subfolders
    for image in train_images:
        src_path = os.path.join(class_path, image)
        dest_path = os.path.join(train_class_path, image)
        shutil.copy(src_path, dest_path)
    
    for image in validation_images:
        src_path = os.path.join(class_path, image)
        dest_path = os.path.join(validation_class_path, image)
        shutil.copy(src_path, dest_path)

print("Data split into train and validation folders.")

# Getting the training data ready.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# training parameters
train_generator = train_datagen.flow_from_directory(
    'path_to/train',
    target_size=(224, 224),  # Adjust size according to your model
    batch_size=16,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'path_to/validation',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)


input_tensor = Input(shape=(224, 224, 3))

# Creating CNN
cnn_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)

x = cnn_model.output

x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x= Dense(1024, activation= 'relu', kernel_initializer='uniform')(x)
predictions = Dense(6, activation='softmax', kernel_initializer='uniform', kernel_regularizer='l2')(x)

model = Model(inputs=input_tensor, outputs=predictions)

print("Model built")
model.summary()

# Model Compling and training
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stop]
)




# This is for TESTING the model on any test image. Uncomment this for testing

# test_generator = validation_datagen.flow_from_directory(
#     'path_to_main_folder/test',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=False
# )

# test_loss, test_acc = model.evaluate(test_generator)
# print("Test accuracy:", test_acc)


# Output
scores = model.evaluate_generator(validation_generator)
print(model.metrics_names, scores)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))