
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

# Define the path to your dataset
dataset_dir = 'C:/Users/shrey/Downloads/Medicinal Leaf Dataset/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images'

# Define constants
image_size = (299, 299)
batch_size = 32

# Create an instance of the InceptionV3 model pre-trained on ImageNet
base_model = InceptionV3(weights='imagenet', include_top=False)

# Create a custom head for classification
head_model = tf.keras.Sequential([
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='softmax')  # Replace '30' with the number of classes in your dataset
])

# Combine the base model and the custom head
model = tf.keras.Model(inputs=base_model.input, outputs=head_model(base_model.output))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators for training and validation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2  # Adjust the validation split as needed
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # Adjust the number of epochs as needed
    verbose=1
)

# Save the trained model
model.save('plant_classification_model.h5')