import os
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications import VGG16

from useful_functions import upsample_model
from metrics import dice_coef, dice_loss
from data_generators import train_val_data
from constants import EPOCH_COUNT, MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

if not os.path.exists('plots'):
    os.mkdir('plots')

def save_model_training_plot(history):
    """
    Function to plot and save training history.

    Parameters:
        history (History): History object containing training history.

    Returns:
        None
    """
     
    # Plot dice score
    plt.plot(history.history['dice_coef'], label = 'train')
    plt.plot(history.history['val_dice_coef'], label = 'validation')
    plt.title('Dice score')
    plt.ylabel('Dice score')
    plt.xlabel('Epoch')
    plt.legend(loc = 'upper left')
    plt.savefig('plots/dice_score_plot.png')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label = 'train')
    plt.plot(history.history['val_loss'], label = 'validation')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc = 'upper right')
    plt.savefig('plots/loss_plot.png')
    plt.show()


def spatial_attention(input_tensor):
    """
    Implements a spatial attention mechanism.

    This mechanism selectively highlights important spatial features within a given input tensor,
    improving the model's ability to focus on relevant information for better performance.

    Parameters:
        input_tensor (Tensor): Input tensor.

    Returns:
        Tensor: Scaled input tensor after applying spatial attention mechanism.
    """
    squeeze = GlobalAveragePooling2D()(input_tensor)
    squeeze = Reshape((1, 1, input_tensor.shape[-1]))(squeeze)
    excitation = Dense(input_tensor.shape[-1] // 2, activation = 'relu')(squeeze)
    excitation = Dense(input_tensor.shape[-1], activation = 'sigmoid')(excitation)
    excitation = Reshape((1, 1, input_tensor.shape[-1]))(excitation)
    scaled_input = Multiply()([input_tensor, excitation])
    return scaled_input

def compile_model():
    """
    Compile a U-Net style convolutional neural network for semantic segmentation.

    Returns:
    - model: Compiled U-Net model.
    """
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
    
    # Apply spatial attention mechanism to each block
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
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv9) # Use sigmoid for binary classification (pixels as mask)

    # Create and compile the model
    model = Model(inputs = base_model.input, outputs = output)

    model.compile(optimizer = 'adam',
                  loss = dice_loss,
                  metrics = dice_coef)
    print('Step 1 finished')

    return model


def train_model(model):
    """
    Trains the given model using specified data generators and callbacks.

    Parameters:
        model: Keras Model
            The model to be trained.

    Returns:
        loss_history: History
            History object containing training loss values and metrics.
        model: Keras Model
            The trained model.
    """

    print('Step 2. Creating data generators')

    aug_gen, val, step_count = train_val_data()

    print('Step 2 finished')

    # callbacks for saving model, learning_rate scheduling and stoping on plateu.
    weight_path = "checkpoints/model.h5"

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)

    early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                        patience=3)

    callbacks_list = [checkpoint, early]


    print('Step 3. Training')

    loss_history = model.fit(aug_gen, steps_per_epoch = step_count, epochs = EPOCH_COUNT, validation_data = (val[0], val[1]), callbacks = callbacks_list)

    print('Step 3 finished')

    save_model_training_plot(loss_history)
    print('Model history was saved in output library. Model was saved in checkpoints')

    upres_model = upsample_model(model)

    return loss_history, upres_model


if __name__ == "__main__":
    model = compile_model()
    model = train_model(model)[1]
