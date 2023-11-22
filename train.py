import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy
from data_generators import create_train_generator, create_validation_generator
from constants import BATCH_SIZE, TRAIN_SAMPLES, VALID_SAMPLES, LEARNING_RATE, EPOCH_COUNT, MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, BatchNormalization, UpSampling2D, Concatenate, GlobalAveragePooling2D, Reshape, Multiply
from tensorflow.keras.applications import VGG16

#function for plotting and saving results
def save_model_training_plot(model):
    # summarize history for dice score
    plt.plot(model.history['dice_coef'])
    plt.plot(model.history['val_dice_coef'])
    plt.title('Dice score')
    plt.ylabel('Dice score')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()

    # summarize history for loss
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper right')

    plt.savefig('../projects/output/train_val_history.png')


# Spatial attention mechanism
def spatial_attention(input_tensor):
    '''implements a spatial attention mechanism for selectively highlight important spatial features within a given input tensor,
       ultimately improving the model's ability to focus on relevant information for better performance in downstream tasks.'''
    squeeze = GlobalAveragePooling2D()(input_tensor)
    squeeze = Reshape((1, 1, input_tensor.shape[-1]))(squeeze)
    excitation = Dense(input_tensor.shape[-1] // 2, activation = 'relu')(squeeze)
    excitation = Dense(input_tensor.shape[-1], activation = 'sigmoid')(excitation)
    excitation = Reshape((1, 1, input_tensor.shape[-1]))(excitation)
    scaled_input = Multiply()([input_tensor, excitation])
    return scaled_input

# function for dice score calculation
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis = [1,2,3])
    union = K.sum(y_true, axis = [1,2,3]) + K.sum(y_pred, axis = [1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis = 0)

# hand-made loss function that combine binary_crossentropy and dice_coef
def dice_p_bce(in_gt, in_pred):
    return (0.001 * binary_crossentropy(in_gt, in_pred)) - dice_coef(in_gt, in_pred)

def compile_model():
    print('Step 1. Compiling model')
    # Encode part
    # I have used pre-trained model for better results
    base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT, 3))

    # Freeze the weights of the VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    # Get the output from different blocks of the VGG16 model
    block1 = base_model.get_layer('block1_conv2').output
    block2 = base_model.get_layer('block2_conv2').output
    block3 = base_model.get_layer('block3_conv3').output
    block4 = base_model.get_layer('block4_conv3').output
    block5 = base_model.get_layer('block5_conv3').output
    
    block1 = spatial_attention(block1)
    block2 = spatial_attention(block2)
    block3 = spatial_attention(block3)
    block4 = spatial_attention(block4)
    block5 = spatial_attention(block5)
    
    # Decoder part
    up1 = UpSampling2D((2, 2))(block5)
    concat1 = Concatenate()([up1, block4])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat1)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up2 = UpSampling2D((2, 2))(conv6)
    concat2 = Concatenate()([up2, block3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat2)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up3 = UpSampling2D((2, 2))(conv7)
    concat3 = Concatenate()([up3, block2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up4 = UpSampling2D((2, 2))(conv8)
    concat4 = Concatenate()([up4, block1])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    # Output layer
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv9) # use sigmoid, because binary classification if pixels is mask.

    # Create and compile the model
    model = Model(inputs = base_model.input, outputs = output)

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = LEARNING_RATE),
                  loss = dice_p_bce,
                  metrics = dice_coef)
    print('Step 1 finished')

    return model


def train_model(model):
    print('Step 2. Creating data generators')

    train_gen = create_train_generator()
    valid_gen = create_validation_generator()
    print('Step 2 finished')

    # callbacks for saving model, learning_rate scheduling and stoping on plateu.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('../projects/checkpoints/best_nn.h5',
                                           save_best_only = True,
                                           monitor = 'val_dice_coef',
                                           mode = 'max'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                             mode = 'min',
                                             factor = 0.2,
                                             patience = 2,
                                             min_lr = 0.000005,
                                             verbose = 1),
        tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 6)
    ]

    train_step_size = TRAIN_SAMPLES // BATCH_SIZE
    valid_step_size = VALID_SAMPLES // BATCH_SIZE

    print('Step 3. Training')

    # fit model on generator
    result = model.fit(x = train_gen,
                        validation_data = valid_gen,
                        validation_steps = valid_step_size,
                        epochs = EPOCH_COUNT,
                        steps_per_epoch = train_step_size,
                        verbose = 1,
                        callbacks = [callbacks])
    print('Step 3 finished')

    save_model_training_plot(result)
    print('Model history was saved in output library. Model was saved in checkpoints')

    return result, model


def evaluate_model(model):
    validation_gen = create_validation_generator()

    print('Evaluating model performance.')

    # evaluate on 200 batches
    scores = model.evaluate(validation_gen, verbose = 1, steps = 200)
    print(f'Loss: {scores[0]}, Dice score: {scores[1] * 100}')

if __name__ == "__main__":
    model = compile_model()
    model = train_model(model)[1]
    evaluate_model(model)
