import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from preprocess import get_image_dataset, get_mask_dataset
from sklearn.model_selection import train_test_split
import cv2
import os
from util import *


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
     
    model =  Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


image_dataset = get_image_dataset(blur=True)
labels = get_mask_dataset(blur=True)

labels = np.expand_dims(labels, axis=3)

# Create train and test splits
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels, test_size = 0.20, random_state = 42)

# testing - filter to only include images with targets (prevent model from always predicting 0)
non_empty_indices = [i for i, m in enumerate(y_train) if np.sum(m) > 0]
filtered_images = X_train[non_empty_indices]
filtered_labels = y_train[non_empty_indices]

# compare_image_sets(filtered_images, filtered_labels)


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]

# define training epochs
epochs = 51

model_name = f'{epochs}_epoch_model.keras'
model_path = os.path.join('models', model_name)

model = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, image_dataset.shape[3])
if not os.path.exists(model_path):
    test = model.fit(X_train, y_train, batch_size=16, verbose=1, epochs=epochs, validation_data = (X_test, y_test), shuffle=False)
    model.save(model_path)


# get all pretrained models for performance evaluation
model_paths = [os.path.join('models', m) for m in os.listdir('models') if m.endswith('.keras')]
performances = {}

# calculate dice and jaccard for each image and each model
for model_path in model_paths[:1]:
    model = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, image_dataset.shape[3])
    model.load_weights(model_path)
    model_dice, model_jaccard = [], []
    zero_dice_count = 0

    for i in range(len(X_test)):
        # shape to (1, 256, 256, 3)
        img = np.expand_dims(X_test[i], axis=0) 
        prediction = np.squeeze((model.predict(img) > 0.75).astype(np.uint8), axis=0)

        # filter prediction to be zero if only a small number of pixels predicted
        threshold = 25
        pixels = prediction.sum()
        if pixels < threshold:
            prediction = np.zeros_like(prediction)

        dice_val = calculate_dice(y_test[i], prediction)
        jaccard_val = calculate_jaccard(y_test[i], prediction)

        model_dice.append(dice_val)
        model_jaccard.append(jaccard_val)

        if dice_val == 0:
            zero_dice_count += 1
        if dice_val < 1 and model_path == 'models/51_epoch_model.keras':
            compare_image_label_prediction(X_test[i], y_test[i], prediction)


    performances[model_path] = (model_dice, model_jaccard)


print(f'Zero dice count: {zero_dice_count}')

plt.figure(figsize=(16, 8))
plt.subplot(1,2,1)

for path in performances.keys():
    dice = performances[path][0]
    x = range(len(dice))
    plt.scatter(x, dice, label = path)
plt.title('Dice For Each Image')
plt.legend()

plt.subplot(1,2,2)
for path in performances.keys():
    jaccard = performances[path][1]
    x = range(len(jaccard))
    plt.scatter(x, jaccard, label = path)
plt.title('Jaccard For Each Image')
plt.legend()

plt.show()

    


