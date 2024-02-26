import os
import pandas as pd
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from constants import TRAIN_IMAGES_PATH, BATCH_SIZE, SEED
from useful_functions import create_mask

mask_train = pd.read_csv('masks.csv') #load dataframe with coded masks
balanced_train_df = pd.read_csv('pretrain_set.csv') #images data


# Splitting data into training and validation sets
train_id, valid_id = train_test_split(balanced_train_df, test_size = 0.2, stratify = balanced_train_df['Ships'], random_state = SEED)

train_df = pd.merge(mask_train, train_id) #ImageId with masks
valid_df = pd.merge(mask_train, valid_id)

def make_image_gen(df):
    """
    Generate batches of images and corresponding masks from DataFrame.
    
    Args:
    df: DataFrame containing image data and corresponding masks.
    
    Yields:
    Tuple: Batch of  cropped and normalazing images and corresponding masks.
    """

    group_id = list(df.groupby('ImageId'))
    np.random.shuffle(group_id)

    image = []
    mask = []

    while True:

        for id, masks in group_id:

            crop_image = Image.open(os.path.join(TRAIN_IMAGES_PATH, id))
            crop_mask = np.expand_dims(create_mask(masks['EncodedPixels'].values), -1)
            crop_image, crop_mask = np.array(crop_image), np.array(crop_mask)
            
            crop_image = crop_image[::3, ::3]
            crop_mask = crop_mask[::3, ::3]

            image += [crop_image]
            mask += [crop_mask]

            if len(image) >= BATCH_SIZE:

                yield np.stack(image, 0) / 255.0, np.stack(mask, 0)
                image, mask=[], []

def datagenerator():
    """
    Create an ImageDataGenerator object for data augmentation.
    
    Returns:
    ImageDataGenerator object for data augmentation.
    """
    image_gen = ImageDataGenerator(rotation_range = 45, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                  data_format = 'channels_last')
                   
    return image_gen

# Create ImageDataGenerator objects for image and mask augmentation
image_gen = datagenerator()
label_gen = datagenerator()

def augmentation_generator(image_mask_gen):
    """
    Augment images and masks using the provided generator.
    
    Args:
    image_mask_gen: Generator yielding batches of images and masks.
    
    Yields:
    Tuple: Augmented images and masks.
    """
    for image, mask in image_mask_gen:

        out_images = image_gen.flow(image, batch_size = image.shape[0], seed = SEED, shuffle=True)
        
        out_masks = label_gen.flow(mask, batch_size = image.shape[0], seed = SEED, shuffle=True)

        yield next(out_images), next(out_masks)

def train_val_data():
    """
    Prepare data for training and validation.
    
    Returns:
    Tuple: Generator for augmented images and masks, validation data (images and masks), and step count.
    """
    aug_gen = augmentation_generator(make_image_gen(train_df))
    valid_x, valid_y = next(make_image_gen(valid_df))

    step_count = train_df.shape[0] // BATCH_SIZE

    return aug_gen, (valid_x, valid_y), step_count