import keras.backend as K

def dice_coef(y_true, y_pred, smooth = 1):
    """
    Dice coefficient metric for model evaluation.

    Parameters:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        Tensor: Dice coefficient value.
    """
    y_true = K.cast(y_true, dtype = 'float32')
    y_pred = K.cast(y_pred, dtype = 'float32')
    intersection = K.sum(y_true * y_pred, axis = [1,2,3])
    union = K.sum(y_true, axis = [1,2,3]) + K.sum(y_pred, axis = [1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis = 0)

def dice_loss(y_true, y_pred):
    """
    Dice loss function for model optimization.

    Parameters:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: Dice loss value.
    """
    return 1 - dice_coef(y_true, y_pred)