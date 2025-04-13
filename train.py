import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.utils import normalize
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from preprocess import get_image_dataset, get_mask_dataset
from sklearn.model_selection import train_test_split
import cv2
import os
from util import *


image_dataset = get_image_dataset()
labels = get_mask_dataset()

"""
for _ in range(50):
    image_number = random.randint(0, len(image_dataset) - 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image_dataset[image_number])
    plt.subplot(122)
    plt.imshow(labels[image_number])
    plt.show()
"""

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model

labels = np.expand_dims(labels, axis=3)
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels, test_size = 0.20, random_state = 42)
print(f'{len(X_train)} training images loaded')

non_empty_indices = [i for i, m in enumerate(y_train) if np.sum(m) > 0]
filtered_images = X_train[non_empty_indices]
filtered_labels = y_train[non_empty_indices]

#compare_image_sets(filtered_images, filtered_labels)


# compare_image_sets(X_train, y_train)
print(f'{len(X_test)} test images loaded')

weights = [0.5, 0.5]
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]

model_name = 'test_model.keras'

model = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, image_dataset.shape[3])
if not os.path.exists('test_model.keras'):
    test = model.fit(filtered_images, filtered_labels, batch_size=16, verbose=1, epochs=25, validation_data = (X_test, y_test), shuffle=False)
    model.save(model_name)
else:
    model.load_weights(model_name)

# Load image
test_img_other = cv2.imread('train_mini/images/image_9.png')  # Default is color
test_img_other = cv2.cvtColor(test_img_other, cv2.COLOR_BGR2RGB)

# Resize if necessary
# REPLACE LATER WITH SLIDING WINDOW TO PREVENT DISTORTION

test_img_other = cv2.resize(test_img_other, (IMG_WIDTH, IMG_HEIGHT))

# Normalize to [0,1]
test_img_other_norm = test_img_other.astype(np.float32) / 255.0

test_img_other_input = np.expand_dims(test_img_other_norm, axis=0)
print(test_img_other.shape)
prediction_other = np.squeeze((model.predict(test_img_other_input) > 0.25).astype(np.uint8), axis=0)


for i in range(len(X_test)):
    # shape to (1, 256, 256, 3)
    img = np.expand_dims(X_test[i], axis=0) 
    prediction = np.squeeze((model.predict(img) > 0.35).astype(np.uint8), axis=0)
    compare_image_label_prediction(X_test[i], y_test[i], prediction)
    j, d = calculate_jaccard_and_dice(img, prediction)
