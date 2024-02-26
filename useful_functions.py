import numpy as np
from skimage.morphology import label
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AvgPool2D, UpSampling2D

def upsample_model(model):
    upres_model = Sequential()
    upres_model.add(AvgPool2D((3, 3), input_shape = (None, None, 3)))
    upres_model.add(model)
    upres_model.add(UpSampling2D((3, 3)))
    upres_model.save('checkpoints/upres_model.h5')

    return upres_model

def rle_decode(mask_rle, shape=(768, 768)):
    '''Decode a run-length encoded mask.

    Parameters: 
       mask_rle: str
           Run-length encoded mask
       shape: tuple, optional
           Shape of the mask array. Default is (768, 768).

    Returns:
       np.array
           Decoded mask as numpy array with 0-1 encoding
    '''
    if not isinstance(mask_rle, str):
        img = np.zeros(shape[0] * shape[1], dtype = np.uint8)
        return img.reshape(shape).T

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype = int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype = np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  


def create_mask(in_mask_list):
    '''Create a mask from a list of run-length encoded masks.

    Parameters: 
       in_mask_list: list
           List of run-length encoded masks

    Returns:
       np.array
           Combined mask as numpy array
    '''
    all_masks = np.zeros((768, 768), dtype = np.uint8)

    for mask in in_mask_list:
        all_masks += rle_decode(mask)
    return all_masks

def rle_encode(img, min_max_threshold=1e-3):
    '''
    Encode a mask as run-length encoding.

    Parameters:
       img: np.array
           Input mask array, where 1 represents the mask and 0 represents background
       min_max_threshold: float, optional
           Minimum threshold to consider encoding. Default is 1e-3.
       max_mean_threshold: float, optional
           Maximum mean threshold for the mask to be encoded. Default is None.

    Returns:
       str
           Run-length encoded mask
    '''
    if np.max(img) < min_max_threshold:
        return '' 
    
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def sep_rle(img, **kwargs):
    '''
    Encode connected regions as separated masks.

    Parameters:
       img: np.array
           Input mask array
       **kwargs: dict
           Additional keyword arguments to pass to `rle_encode`

    Returns:
       list
           List of run-length encoded masks
    '''
    labels = label(img)

    return [rle_encode(np.sum(labels== k, axis = 2), **kwargs) for k in np.unique(labels[labels > 0])]