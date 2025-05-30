import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from preprocess import preprocesser

class classifier:
    def __init__(self, height=1024, width=1024, channels=3):
        self.IMAGE_HEIGHT = height
        self.IMAGE_WIDTH = width
        self.IMAGE_CHANNELS = channels
        self.model = self.classifier_model()
    
    def load_data(self, path_to_data_folder):
        """
        Call the preprocessor class to load the image and mask data
        Convert masks to T/F based on whether a positive pixel exists or not
        """
        data_processor = preprocesser(path_to_data_folder)
        
        images = data_processor.get_image_dataset()
        masks = data_processor.get_mask_dataset()

        # raise error if no data loaded
        if len(images) == 0 or len(masks) == 0:
            raise FileNotFoundError(f"""No data loaded. Make sure the passed directory
                                     {path_to_data_folder} exists and contains data""")

        # If classifier was initialized without specifying image shape, update image shapes
        if not self.IMAGE_HEIGHT:
            self.IMAGE_HEIGHT = images[0].shape[0]
        if not self.IMAGE_WIDTH:
            self.IMAGE_WIDTH = images[0].shape[1]
        if not self.IMAGE_CHANNELS:
            self.IMAGE_CHANNELS = images[0].shape[2]

        labels = [1 if np.max(mask) > 0 else 0 for mask in masks]

        return images, labels
    
    def classifier_model(self):
        """
        Expected input: 1024 x 1024 x 3. If using on different size, may want to adjust
        layer values for efficiency and accuracy
        """
        input_shape = (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS)
        model = models.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2),  # 512x512

            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2),  # 256x256

            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2),  # 128x128

            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2),  # 64x64

            layers.Conv2D(512, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
c = classifier()
images, labels = c.load_data('train_mini')
print(labels)


