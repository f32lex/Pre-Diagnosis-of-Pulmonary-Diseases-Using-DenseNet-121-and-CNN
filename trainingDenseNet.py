from keras.models import Sequential, Model  # Added Sequential here
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications.densenet import preprocess_input
from keras.models import load_model
import numpy as np
from skimage import exposure  # Ensure you have scikit-image installed for this

# Image dimensions
img_width, img_height = 224, 224

# Paths
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# Model parameters
nb_train_samples = 2000  # Total number of training images
nb_validation_samples = 500  # Total number of validation images
epochs = 25  # Moderate number of epochs
batch_size = 32  # A reasonable batch size

# Early stopping and learning rate reduction
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.00001)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dropout(0.5),
    Dense(1),
    Activation('sigmoid')
])

# Load DenseNet-121 with pre-trained ImageNet weights
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Adding custom layers for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling
x = Dense(1024, activation='relu')(x)  # Dense layer
predictions = Dense(1, activation='sigmoid')(x)  # Final output layer for binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#for layer in base_model.layers:
 #   layer.trainable = False
    
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    
    # Convert the image to float and rescale it to the range [0, 1]
    img = img_as_float(img)

    # Apply adaptive histogram equalization
    img = exposure.equalize_adapthist(img)
    
    # Normalize the image as expected by DenseNet
    img = preprocess_input(img)
    
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# Update ImageDataGenerator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Use DenseNet specific preprocessing
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)  # Use DenseNet specific preprocessing

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),  # Resize images
    batch_size=batch_size,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),  # Resize images
    batch_size=batch_size,
    class_mode='binary')

# Automatically calculate the steps per epoch and validation steps
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

# Train the model
model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stop, reduce_lr])

# Save the model
model.save('model.keras')
model.save_weights('weights.keras')
