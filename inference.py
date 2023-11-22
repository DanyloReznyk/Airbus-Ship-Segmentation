import random
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import click
from constants import ORIG_IMG_WIDTH, ORIG_IMG_HEIGHT, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH, TEST_IMAGES_PATH, TRAIN_IMAGES_PATH, TRAINED_MODEL_PATH, SUBMISSION_DF_PATH

# create folders, it they don't exist
if not os.path.exists('../projects/output/validation'):
    os.mkdir('../projects/output/validation')
    
if not os.path.exists('../projects/output/inference'):
    os.mkdir('../projects/output/inference')
    

# loads pre-trained model from pre-defined folder
def load_model():
    print('Loading pre-trained model.')

    model = tf.keras.models.load_model(TRAINED_MODEL_PATH, compile = False)

    print('Successfully loaded pre-trained model.')
    return model


# image prediction function
def custom_predict(model, img_name, test_data = True):

    if test_data:
        img = tf.keras.utils.load_img(os.path.join(TEST_IMAGES_PATH, img_name))
    else:
        img = tf.keras.utils.load_img(os.path.join(TRAIN_IMAGES_PATH, img_name))

    img = tf.image.resize(img, [MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT], preserve_aspect_ratio = True)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # predict
    result = model.predict(img_array)
    # leave prediction with more then 50% probability
    result = result > 0.5

    result = np.squeeze(result, axis=0)

    return result

def rle_decode(mask_rle, shape = (ORIG_IMG_WIDTH, ORIG_IMG_HEIGHT)):
    '''Parameters: 
       mask_rle: run-length in string
       shape: tuple of height and width of array
       return: np.array with 0-1 encoding
      '''
    # returns empty mask if row has no data
    if not isinstance(mask_rle, str):
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        return img.reshape(shape).T

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# the same function, as in EDA
def create_mask(df, img_name):
    masks = df['EncodedPixels'].loc[df['ImageId'] == img_name].to_list()

    final_mask = np.zeros((768, 768))

    # iterate through masks array, decode each RLE string
    for mask in masks:
        final_mask += rle_decode(mask)

    return final_mask

# compare prediction to truth
def compare_to_truth(model, compare_to_t = False):
    if compare_to_t:
        print('Visualizing validation prediction results.')
        num_predictions = 8

        # read csv to get segmentation masks
        df = pd.read_csv('../projects/input/train_ship_segmentations_v2.csv')

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 10))

        # pick random pictures from train folder
        images = os.listdir(TRAIN_IMAGES_PATH)
        images = random.sample(images, num_predictions)

        for img_name in images:
            # load image
            img = tf.keras.utils.load_img(os.path.join(TRAIN_IMAGES_PATH, img_name))
            ax1.imshow(img)
            ax1.set_title('Image: ' + img_name)

            predicted_mask = custom_predict(model, img_name=img_name, test_data=False)
            ax2.imshow(predicted_mask)
            ax2.set_title('Prediction Masks')

            truth = create_mask(df, img_name)
            ax3.imshow(truth)
            ax3.set_title('Ground Truth')

            fig.savefig(f'../projects/output/validation/{img_name}', bbox_inches='tight', pad_inches=0.1)

        print('Saved to validation folder')


# visualize prediction
def visualize_random_image_inference(model, visualize_inference = False):
    if visualize_inference:
        print('Visualizing inference results.')

        num_predictions = 8
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

        # pick random pictures from test folder
        images = os.listdir(TEST_IMAGES_PATH)
        images = random.sample(images, num_predictions)

        for img_name in images:
            img = tf.keras.utils.load_img(os.path.join(TEST_IMAGES_PATH, img_name))
            ax1.imshow(img)
            ax1.set_title('Image: ' + img_name)

            predicted_mask = custom_predict(model, img_name=img_name)
            ax2.imshow(predicted_mask)
            ax2.set_title('Prediction Masks')

            fig.savefig(f'../projects/output/inference/{img_name}', bbox_inches = 'tight', pad_inches = 0.1)

        print('Saved inference visualization to results folder.')


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def predict_all_images(model, predict_all=False):
    if predict_all:
        # read csv and extract image names
        df = pd.read_csv('../projects/input/sample_submission_v2.csv')
        images = df['ImageId'].values

        print('Started predicting test images.')
        for img_name in images:
            mask = custom_predict(model, img_name=img_name)
            mask = np.rot90(mask, k=3)
            mask = np.fliplr(mask)
            mask_rle = rle_encode(mask)

            # update row value with RLE string
            df.loc[df['ImageId'] == img_name, ['EncodedPixels']] = mask_rle
        # save to csv
        df.to_csv('../projects/output/submission.csv', index=False)
        print('Saved prediction results in output folder, submission.csv')

def show_submission_images(show_submission=False):
    if show_submission:
        df = pd.read_csv('../projects/output/submission.csv')
        rows = df.sample(20).values

        for row in rows:
            img_name = row[0]
            rle = row[1]

            img = tf.keras.utils.load_img(TEST_IMAGES_PATH)

            mask = rle_decode(rle, shape = (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT))

            # create subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 10))

            ax1.axis('off')
            ax2.axis('off')
            ax1.imshow(img)
            ax1.set_title(f'Original image', fontsize = 15)

            ax2.imshow(mask)
            ax2.set_title('Mask image', fontsize = 15)

            fig.savefig(f'../projects/output/inference/{img_name}', bbox_inches = 'tight', pad_inches = 0.1)

        print('Saved visualization to inference folder.')


@click.command()
@click.option('--compare_to_t', is_flag = True, help = 'Predict on validation images and compare to ground truth.')
@click.option('--visualize_inference', is_flag = True, help = 'Predict on test images and save images.')
@click.option('--predict_all', is_flag = True, help = 'Predict all test images and save RLE information to csv file.')
@click.option('--show_submission', is_flag = True, help = 'Saves decoded RLE images from submission csv into folder.')

def main(compare_to_t, visualize_inference, predict_all, show_submission):
    model = load_model()
    compare_to_truth(model, compare_to_t)
    visualize_random_image_inference(model, visualize_inference)
    predict_all_images(model, predict_all)
    show_submission_images(show_submission)


if __name__ == '__main__':
    main()