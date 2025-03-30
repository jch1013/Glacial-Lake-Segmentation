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
root_dir = '/Users/jacksonhayward/Desktop/Glacial-Lake-Segmentation/'
patch_size = 256

def get_image_dataset():
    image_dataset = []  
    for path, subdirs, files in os.walk(root_dir):
        dirname = path.split(os.path.sep)[-1]

def get_mask_dataset():
    pass


get_image_dataset()


for path in os.listdir('miniset'):
    dir = os.path.join('miniset', path)
    try:
        image = [p for p in os.listdir(dir) if p.endswith('.tif')][0]
        #os.rename()
        old_image_path = os.path.join(dir, image)
        new_image_path = os.path.join('miniset', 'masks', path)
        os.rename(old_image_path, new_image_path)
    except:
        print(f'{dir} could not be read')