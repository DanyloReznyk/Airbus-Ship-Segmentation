import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from constants import TRAIN_DF_PATH, TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, TEST_IMAGES_PATH, \
    BATCH_SIZE, MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT, SEED

df_train = pd.read_csv(TRAIN_DF_PATH) # load train data for generators

'''This code initializes two ImageDataGenerator objects for data augmentation
The generators apply rescaling, rotation, horizontal and vertical flips, and zooming.
Additionally, a validation split of 20% is specified'''

def create_data_generator():
    return ImageDataGenerator(
        rescale = 1. / 255,
        rotation_range = 90,
        horizontal_flip = True,
        vertical_flip = True,
        zoom_range = 0.1,
        validation_split = 0.2
    )

image_datagen = create_data_generator()

mask_datagen = create_data_generator()


# Function to create a training data generator with augmented images and masks
def create_train_generator():
    # Generate image and mask batches for training from the specified dataframe and directories
    train_image_gen = image_datagen.flow_from_dataframe(dataframe = df_train,
                                                        directory = TRAIN_IMAGES_PATH,
                                                        x_col = 'ImageId',
                                                        class_mode = None,
                                                        target_size = (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT),
                                                        batch_size = BATCH_SIZE,
                                                        subset = 'training',
                                                        shuffle = False,
                                                        seed = SEED)

    train_mask_gen = mask_datagen.flow_from_dataframe(dataframe = df_train,
                                                      directory = TRAIN_MASKS_PATH,
                                                      x_col = 'ImageId',
                                                      class_mode = None,
                                                      target_size = (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT),
                                                      batch_size = BATCH_SIZE,
                                                      subset = 'training',
                                                      shuffle = False,
                                                      color_mode = 'grayscale',
                                                      seed = SEED)
    
    # Combine image and mask generators into a single generator using zip
    train_generator = zip(train_image_gen, train_mask_gen)

    return train_generator

# Function to create a validation data generator with augmented images and masks
def create_validation_generator():
    # Generate image and mask batches for validation from the specified dataframe and directories
    valid_image_gen = image_datagen.flow_from_dataframe(dataframe = df_train,
                                                        directory = TRAIN_IMAGES_PATH,
                                                        x_col = 'ImageId',
                                                        class_mode = None,
                                                        target_size = (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT),
                                                        batch_size = BATCH_SIZE,
                                                        subset = 'validation',
                                                        shuffle = False,
                                                        seed = SEED)

    valid_masks_gen = mask_datagen.flow_from_dataframe(dataframe = df_train,
                                                       directory = TRAIN_MASKS_PATH,
                                                       x_col = 'ImageId',
                                                       class_mode = None,
                                                       target_size = (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT),
                                                       batch_size = BATCH_SIZE,
                                                       subset = 'validation',
                                                       shuffle = False,
                                                       color_mode = 'grayscale',
                                                       seed = SEED)
    
    # Combine image and mask generators into a single generator using zip
    valid_generator = zip(valid_image_gen, valid_masks_gen)

    return valid_generator

# Function to create a test data generator for evaluation
def create_test_generator():
    # Generate test data batches with rescaled images from the specified directory
    test_datagen = ImageDataGenerator(rescale = 1. / 255)

    test_generator = test_datagen.flow_from_directory(directory = TEST_IMAGES_PATH,
                                                      classes = ['test_v2'],
                                                      batch_size = 1,
                                                      seed = SEED,
                                                      shuffle = True,
                                                      target_size = (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT))

    return test_generator