import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.models import load_model
from preprocess import preprocesser
from sklearn.model_selection import train_test_split
import os
from util import *


class unet_segmentation_model:

    def __init__(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        self.model = self.create_model()

    def create_model(self):

        inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
        s = inputs

        #Contraction path
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        # c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)
        
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        # c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)
        
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        # c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)
        
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        # c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        # c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        
        #Expansive path 
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        # c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        
        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        #c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
        
        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        #c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        
        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        #c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        
        model =  Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
        return model

    def train_model(self, model_name, training_directory, epochs):
        loader = preprocesser(training_directory)
        image_dataset = loader.get_image_dataset()
        labels_dataset = loader.get_mask_dataset()
        labels = np.expand_dims(labels_dataset, axis=3)

        # Create train and test splits
        X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels, test_size = 0.4, random_state = 42)

        # Verify that the model name has the .keras extension, and if not, add it
        if not model_name.endswith('.keras'):
            model_name = f'{model_name}.keras'
        model_path = os.path.join('models', model_name)

        if not os.path.exists(model_path):
            test = self.model.fit(X_train, y_train, batch_size=1, verbose=1, epochs=epochs, validation_data = (X_test, y_test), shuffle=False)
            self.model.save(model_path)
        else:
            print(f'The model {model_name} already exists. Loading model.')
            self.model = load_model(model_path)

        self.test_model(X_test, y_test)

    def test_model(self, X_test, y_test):
        for i in range(len(X_test)):
            # shape to (1, 256, 256, 3)
            img = np.expand_dims(X_test[i], axis=0)

            # Predict the pixels that contain glacial lake - currently using 75% liklihood
            prediction = np.squeeze((self.model.predict(img) > 0.3).astype(np.uint8), axis=0)
            print(f"Prediction min: {prediction.min()}, max: {prediction.max()}")

            # filter prediction to be zero if only a small number of pixels predicted
            threshold = 0
            pixels = prediction.sum()
            if pixels < threshold:
                prediction = np.zeros_like(prediction)

            dice_val = calculate_dice(y_test[i], prediction)
            jaccard_val = calculate_jaccard(y_test[i], prediction)

            compare_image_label_prediction(X_test[i], y_test[i], prediction)
            compare_images(X_test[i], prediction)

            print(f'Dice score:    {dice_val}')
            print(f'Jaccard score: {jaccard_val}')



test_model = unet_segmentation_model(1024, 1024, 3)
test_model.train_model('overfit_test', 'train_mini', 25)

    


