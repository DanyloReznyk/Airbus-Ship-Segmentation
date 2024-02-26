import os
import click
import gdown
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.cli import tqdm
from skimage.io import imread
from skimage.morphology import binary_opening, disk

from useful_functions import create_mask, sep_rle

# # create folders, it they don't exist
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')    

if not os.path.exists('inference'):
    os.mkdir('inference')
    
def predict(model, img):
    """
    Predicts the segmentation mask for a single image.

    Args:
        model (tf.keras.Model): Pre-trained segmentation model.
        img (str): Image filename.
        path (str): Path to the directory containing test images.

    Returns:
        tuple: Tuple containing the predicted segmentation mask and the preprocessed image.
    """
    TEST_IMAGES_PATH = os.path.join(args.directory, 'test_v2')

    image = imread(os.path.join(TEST_IMAGES_PATH, img))
    image = np.expand_dims(image, 0) / 255.0
    mask = model.predict(image)[0]
    return mask

def smoothing_predict(model, img):
    """
    Predicts and smoothes the segmentation mask for a single image.

    Args:
        model (tf.keras.Model): Pre-trained segmentation model.
        img (str): Image filename.
        path (str): Path to the directory containing test images.

    Returns:
        tuple: Tuple containing the smoothed segmentation mask and the preprocessed image.
    """
     
    mask = predict(model, img)
    return binary_opening(mask > 0.99, np.expand_dims(disk(2), -1))

def pred_encode(model, img, **kwargs):
    """
    Encodes predictions for submission.

    Args:
        model (tf.keras.Model): Pre-trained segmentation model.
        img (str): Image filename.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of image and corresponding run-length encoded segmentation masks.
    """

    mask = smoothing_predict(model, img)
    conv_rle = sep_rle(mask, **kwargs)
    return [[img, rle] for rle in conv_rle if rle]



def download_model(download=False):
    """
    Download the model from Google Drive if specified.

    Parameters:
    download (bool): Whether to download the model or not. Default is False.

    Returns:
    None
    """
    if download:
        file_id = '1n1Odif2tG8e_89scVzTDgKe0e7pyy5s_'
        destination = 'checkpoints/upres_model.h5'

        url1 = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url1, destination, quiet=False)

# loads pre-trained model from pre-defined folder
def load_model():
    print('Loading pre-trained model.')

    model = tf.keras.models.load_model('checkpoints/upres_model.h5', compile = False)

    print('Successfully loaded pre-trained model.')
    return model

def predict_all_images(model):
    TEST_IMAGES_PATH = os.path.join(args.directory, 'test_v2')
    SUBMISSION_DF_PATH = os.path.join(args.directory, 'sample_submission_v2.csv')

    # read csv and extract image names
    test_paths = np.array(os.listdir(TEST_IMAGES_PATH))

    predictions = []
    for img_name in tqdm(test_paths):
        predictions += pred_encode(model, img_name, min_max_threshold = 1.0)

    submission = pd.DataFrame(predictions)
    submission.columns = ['ImageId', 'EncodedPixels']
    submission = submission[submission.EncodedPixels.notnull()]

    temporary_df = pd.read_csv(SUBMISSION_DF_PATH)
    temporary_df = pd.DataFrame(np.setdiff1d(temporary_df['ImageId'].unique(), submission['ImageId'].unique(), assume_unique=True), columns = ['ImageId'])
    temporary_df['EncodedPixels'] = None

    submission = pd.concat([submission, temporary_df])

    submission.to_csv('submission.csv', index = False)

    print('Prediction was successful')

def show_submission_images(n = 10):

    TEST_IMAGES_PATH = os.path.join(args.directory, 'test_v2')
    df = pd.read_csv('submission.csv')
    ids = df['ImageId']

    for row in ids:
        if n < 0:
            break

        img_name = row
        
        img = tf.keras.utils.load_img(os.path.join(TEST_IMAGES_PATH, img_name))

        mask = create_mask(df[df['ImageId']==img_name]['EncodedPixels'])
            
        # create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 10))

        ax1.axis('off')
        ax2.axis('off')
        ax1.imshow(img)
        ax1.set_title(f'Original image', fontsize = 15)

        ax2.imshow(mask)
        ax2.set_title('Mask image', fontsize = 15)

        fig.savefig(f'inference/{img_name}', bbox_inches = 'tight', pad_inches = 0.1)

        n -= 1

    print('Saved visualization to inference folder.')


def main():
    download_model(True)
    model = load_model()

    predict_all_images(model)
    show_submission_images()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on images in a directory')
    parser.add_argument('directory', type=str, help='Path to the directory containing images')
    
    # Parse command-line arguments
    args = parser.parse_args()

    main()