import json
import cv2
import numpy as np
from skimage import measure
import simplekml
from tensorflow.keras.models import load_model
import os
from util import compare_images
from PIL import Image
from preprocess import preprocesser
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage

class exporter:
    def __init__(self, input_directory, output_path, model_path):
        self.input = input_directory
        self.output = output_path
        self.model = load_model(model_path)
        self.kml = simplekml.Kml()
        self.preprocessor = preprocesser(input_directory)
        self.scaler = MinMaxScaler()


    def predict_mask(self, image):
        """Make a prediction for the output based on the passed image using the traied model"""
        image = np.expand_dims(image, axis=0)
        mask = np.squeeze((self.model.predict(image) > 0.5).astype(np.uint8), axis=0)
        return mask
    
    def postprocess_mask(self, mask):
        # remove batching dimension
        mask = mask.squeeze(axis=-1)
        # clean up noisy regions
        mask = (mask > 0).astype(np.uint8)
        labeled_mask, num_features = ndimage.label(mask)
        component_sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
        too_small = component_sizes < 500

        # Create a mask of labels to remove
        remove_labels = np.arange(1, num_features + 1)[too_small]

        # Make a copy to clean
        cleaned_mask = mask.copy()

        # Set all pixels belonging to small components to 0
        for label in remove_labels:
            cleaned_mask[labeled_mask == label] = 0
        return cleaned_mask
    
    def load_image(self, image_path):
        image = cv2.imread(image_path, 1)  #Read each image as BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = np.array(image)

        # apply blur if needed
        image = cv2.GaussianBlur(image, (3, 3), 0)

        scaled_image = self.scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
        scaled_image = scaled_image.astype(np.float32)
        return scaled_image
    
    def write_kml(self):
        # read data from the json file generated during data downloading
        json_path = os.path.join(self.input, 'images.json')
        with open(json_path, 'r') as f:
            entries = json.load(f)

        predictions = 0
        for entry in entries:
            # extract data from json file
            image_path = entry['image_path']
            lon_min, lat_min = entry['bbox_lower_left']
            lon_max, lat_max = entry['bbox_upper_right']

            # extract image name from the path in case path has moved
            image_name = image_path.split('/')[-1]
            image_path = os.path.join(self.input, image_name)

            # load the image
            image = self.load_image(image_path)

            # predict the mask
            mask = self.predict_mask(image)
            clean_mask = self.postprocess_mask(mask)

            predictions += 1
            print(f'Predicted {predictions}/{len(entries)}')

            #if np.max(mask) == 1:
                #compare_images(image, clean_mask)

            # transform mask from 3d to 2d, dropping grayscale dimension
            height, width = clean_mask.shape

            # find contours
            contours = measure.find_contours(clean_mask, 0.5)
            for contour in contours:
                coords = []
                for y, x in contour:
                    # convert all pixels along extracted contour into coordinate values
                    lon = lon_min + (x / width) * (lon_max - lon_min)
                    lat = lat_max - (y / height) * (lat_max - lat_min)
                    coords.append((lon, lat))
                if coords:
                    # 5. Add the polygon to the KML
                    self.kml.newpolygon(outerboundaryis=coords)
        
        self.kml.save(self.output)
            

test_input_directory = '/Users/jacksonhayward/Desktop/satellite_images/north_cascades_north'
test_filepath = '/Users/jacksonhayward/Desktop/satellite_images/north_cascades_north/ncn.kml'
test_model_path = '/Users/jacksonhayward/Desktop/Glacial-Lake-Segmentation/models/full_model_test.keras'

test_export = exporter(test_input_directory, test_filepath, test_model_path)
test_export.write_kml()