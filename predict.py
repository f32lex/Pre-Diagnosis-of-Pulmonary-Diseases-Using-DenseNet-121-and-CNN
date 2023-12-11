from keras.models import load_model
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.densenet import preprocess_input
from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Update dimensions for DenseNet-121
img_width, img_height = 224, 224

# Function to preprocess individual images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Load model and print summary
test_model = load_model('model.keras')
test_model.summary()  # This will help you to identify the layer names

# Getting the last dense layer
last_dense_layer_name = 'dense_2'
if last_dense_layer_name in [layer.name for layer in test_model.layers]:
    final_dense = test_model.get_layer(last_dense_layer_name)
    if len(final_dense.get_weights()) > 0:
        weights_dense = final_dense.get_weights()[0]
    else:
        raise ValueError("No weights found in the final dense layer")
else:
    raise ValueError("Layer name not found in the model")

# Getting the last conv layer
last_conv_layer_name = 'conv5_block16_2_conv'
if last_conv_layer_name in [layer.name for layer in test_model.layers]:
    last_conv_layer = test_model.get_layer(last_conv_layer_name)
else:
    raise ValueError("Conv layer name not found in the model")
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.densenet import preprocess_input
from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Update dimensions for DenseNet-121
img_width, img_height = 224, 224

# Function to preprocess individual images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Load model and print summary
test_model = load_model('model.keras')
test_model.summary()  # This will help you to identify the layer names

# Getting the last dense layer
# You need to replace 'last_dense_layer_name' with the actual name of the layer
last_dense_layer_name = 'dense_2'
if last_dense_layer_name in [layer.name for layer in test_model.layers]:
    final_dense = test_model.get_layer(last_dense_layer_name)
    if len(final_dense.get_weights()) > 0:
        weights_dense = final_dense.get_weights()[0]
    else:
        raise ValueError("No weights found in the final dense layer")
else:
    raise ValueError("Layer name not found in the model")

# Getting the last conv layer
# Replace 'last_conv_layer_name' with the name of your last convolutional layer
last_conv_layer_name = 'conv5_block16_2_conv'
if last_conv_layer_name in [layer.name for layer in test_model.layers]:
    last_conv_layer = test_model.get_layer(last_conv_layer_name)
else:
    raise ValueError("Conv layer name not found in the model")

# Image data path
base_dir = "data/test/"


# Prediction function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def predict(image_path, model, threshold=0):
    x = preprocess_image(image_path)
    preds = model.predict(x)
    predicted_class = 'PULMONARY' if preds[0][0] > threshold else 'NORMAL'
    print(f'Predicted Class: {predicted_class} - Probability: {preds[0][0]}')
    return predicted_class, preds

# Loop over test images in both categories
for category in ['1', '2']:
    cat_dir = os.path.join(base_dir, category)
    if os.path.isdir(cat_dir):
        print(f"Processing Category: {category} - {'PULMONARY' if category == '1' else 'NORMAL'}")
        for img_file in os.listdir(cat_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(cat_dir, img_file)
                print(f'Testing Image: {image_path}')
                predict(image_path, test_model)
                print(' ')
