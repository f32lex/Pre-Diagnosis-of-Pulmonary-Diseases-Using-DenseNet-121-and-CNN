import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

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

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Image preprocessing function
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.  # Normalize the image
    return img


# Custom generator
def custom_generator(directory, target_size, batch_size, class_mode):
    categories = ['1', '2']  # Categories in your dataset
    while True:
        batch_x = []
        batch_y = []
        for category in categories:
            cat_dir = os.path.join(directory, category)
            if os.path.isdir(cat_dir):
                filenames = os.listdir(cat_dir)
                selected_files = np.random.choice(filenames, batch_size // len(categories))
                for filename in selected_files:
                    img_path = os.path.join(cat_dir, filename)
                    img = preprocess_image(img_path, target_size)
                    batch_x.append(img[0])
                    batch_y.append(0 if category == '1' else 1)  # 0 for PULMONARY, 1 for NORMAL

        yield np.array(batch_x), np.array(batch_y)


# Data generators for training and validation
train_generator = custom_generator(train_data_dir, (img_width, img_height), batch_size, 'binary')
validation_generator = custom_generator(validation_data_dir, (img_width, img_height), batch_size, 'binary')

# Fit the model
model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[early_stop, reduce_lr]
)

# Save the model and weights
model.save('model.keras')

print("Model trained and saved successfully.")