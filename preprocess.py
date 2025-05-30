import os
import cv2
import numpy as np
from PIL import Image
os.environ["SM_FRAMEWORK"] = "tf.keras"
from sklearn.preprocessing import MinMaxScaler


class preprocesser:
    def __init__(self, root_directory):
        self.root_dir = root_directory
        self.scaler = MinMaxScaler()

    def generate_empty_mask(self, path_to_image):
        if path_to_image.endswith('.DS_Store'):
            return
        image_filename = path_to_image.split(os.path.sep)[-1]
        image_filename = image_filename.split('.')[0]
        mask_filename = os.path.join(self.root_dir, 'masks', image_filename + '.tif')
        mask_filename = mask_filename.replace('image', 'mask')
        if os.path.exists(mask_filename): return
        print(path_to_image)
        image = cv2.imread(path_to_image)
        height, width, channels = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.imwrite(mask_filename, mask)


    def fill_missing_masks(self):
        image_dir = os.path.join(self.root_dir, 'images')
        mask_dir = os.path.join(self.root_dir, 'masks')

        image_dir_files = os.listdir(image_dir)
        for image in image_dir_files:
            mask_name = image.replace('.png', '.tif')
            mask_name = mask_name.replace('image', 'mask')
            if not os.path.exists(os.path.join(mask_dir, mask_name)):
                image_path = os.path.join(image_dir, image)
                self.generate_empty_mask(image_path)

    def get_image_dataset(self, blur=True):
        """
        Loads images in the /images folder within the passed root directory, and standardizes images. 
        Returns a numpy array of images
        
        """
        image_dataset = []
        for path, subdirs, files in os.walk(self.root_dir):
            dirname = path.split(os.path.sep)[-1]
            if dirname == 'images':   #Find all 'images' directories
                images = sorted(os.listdir(path))
                for i, image_name in enumerate(images):  
                    if image_name.endswith(".png"):   #Only read png images...
                    
                        image_path = path+"/"+image_name
                        image = cv2.imread(image_path, 1)  #Read each image as BGR
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(image)
                        image = np.array(image)

                        # apply blur if needed
                        if blur:
                            image = cv2.GaussianBlur(image, (3, 3), 0)

                        scaled_image = self.scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
                        scaled_image = scaled_image.astype(np.float32)
                        image_dataset.append(scaled_image)
                        print(f'Loaded image {image_path}')
        return np.array(image_dataset)
    
    def get_mask_dataset(self, blur=True):
        self.fill_missing_masks()
        mask_dataset = []  
        for path, subdirs, files in os.walk(self.root_dir):
            dirname = path.split(os.path.sep)[-1]
            if dirname == 'masks':   #Find the '/masks' folder
                masks = sorted(os.listdir(path))
                for i, mask_name in enumerate(masks):  
                    if mask_name.endswith(".tif"):   #Only read png images... (masks in this dataset)
                    
                        mask = cv2.imread(path+"/"+mask_name, 0)  #Read each image as Grey (or color but remember to map each color to an integer)
                        mask = mask // 255
                        mask = np.array(mask) 

                        # apply blur if needed
                        if blur:
                            mask = cv2.GaussianBlur(mask, (3, 3), 0)     
                        print(mask.shape, mask.dtype, np.unique(mask))
       
            
                        mask_dataset.append(mask)
        return np.array(mask_dataset)
    

