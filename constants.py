SEED = 1811

# Path variables
TRAIN_DF_PATH = 'input/training_dataframe.csv'
TRAIN_IMAGES_PATH = 'input/train_v2/'
TRAIN_MASKS_PATH = 'output/masks_v2/'
TEST_IMAGES_PATH = 'input/test_v2'
TRAINED_MODEL_PATH = 'checkpoints/best_nn.h5'
SUBMISSION_DF_PATH = 'input/sample_submission_v2.csv'

# Image variables
ORIG_IMG_WIDTH = 768
ORIG_IMG_HEIGHT = 768
MODEL_IMG_WIDTH = 256
MODEL_IMG_HEIGHT = 256

# Model variables
EPOCH_COUNT = 20 # if i had more computional resourse, i do more epoches.
BATCH_SIZE = 10
TRAIN_SAMPLES = 4000 #the same reason, we have too many images, so i decide train for sample.
VALID_SAMPLES = 1000
LEARNING_RATE = 0.0005