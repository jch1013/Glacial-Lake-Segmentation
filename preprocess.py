import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
patch_size = 256
mask_dir = 'train_mini/masks'
data_dir = 'train_mini'

def generate_empty_mask(path_to_image):
    image_filename = path_to_image.split(os.path.sep)[-1]
    image_filename = image_filename.split('.')[0]
    mask_filename = os.path.join(mask_dir, image_filename + '.tif')
    if os.path.exists(mask_filename): return
    image = cv2.imread(path_to_image)
    height, width, channels = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.imwrite(mask_filename, mask)

def fill_masks(data_dir):
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')

    image_dir_files = os.listdir(image_dir)
    for image in image_dir_files:
        mask_name = image.replace('.png', '.tif')
        if not os.path.exists(os.path.join(mask_dir, mask_name)):
            image_path = os.path.join(image_dir, image)
            generate_empty_mask(image_path)

def get_image_dataset():
    image_dataset = []
    data_dir = 'train_mini'
    for path, subdirs, files in os.walk(data_dir):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':   #Find all 'images' directories
            images = os.listdir(path)  #List of all image names in this subdirectory
            for i, image_name in enumerate(images):  
                if image_name.endswith(".png"):   #Only read png images...
                
                    image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
                    SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                    SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                    image = Image.fromarray(image)
                    image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                    image = np.array(image)             
        
                    #Extract patches from each image
                    print("Now patchifying image:", path+"/"+image_name)
                    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
            
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            
                            single_patch_img = patches_img[i,j,:,:]
                            
                            #Use minmaxscaler instead of just dividing by 255. 
                            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                            
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. 
                            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                            image_dataset.append(single_patch_img)
    return image_dataset

def get_mask_dataset():
    mask_dataset = []  
    for path, subdirs, files in os.walk(data_dir):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks':   #Find all 'images' directories
            masks = os.listdir(path)  #List of all image names in this subdirectory
            for i, mask_name in enumerate(masks):  
                if mask_name.endswith(".png"):   #Only read png images... (masks in this dataset)
                
                    mask = cv2.imread(path+"/"+mask_name, 1)  #Read each image as Grey (or color but remember to map each color to an integer)
                    SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                    SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                    mask = np.array(mask)             
        
                    #Extract patches from each image
                    print("Now patchifying mask:", path+"/"+mask_name)
                    patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
            
                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            
                            single_patch_mask = patches_mask[i,j,:,:]
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                            single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
                            mask_dataset.append(single_patch_mask)
    return mask_dataset


#data = get_image_dataset()
#print(len(data))

#generate_empty_mask('train_mini/images/image_10.png')
fill_masks(data_dir)