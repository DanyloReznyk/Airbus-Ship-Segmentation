# This isn't finished solution that had to improve score, but I haven't finished it
# so if you interested in, you can evaluate idea.


'''import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 as VGG16Model
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model 
from constants import TRAIN_IMAGES_PATH, TEST_IMAGES_PATH

DATA_FOLDER_PATH = 'input/'

df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, 'train_ship_segmentations_v2.csv'))

df['exist_ship'] = df['EncodedPixels'].fillna(0)
df.loc[df['exist_ship'] != 0, 'exist_ship'] = 1

train_gp = df.groupby('ImageId').sum().reset_index()
train_gp.loc[train_gp['exist_ship'] > 0,'exist_ship'] = 1
del train_gp['EncodedPixels']
del train_gp['Ships']

train_gp = train_gp.sort_values(by = 'exist_ship')
train_gp = train_gp.drop(train_gp.index[0:100000])

train_sample = train_gp.sample(5000)

training_img_data = []
target_data = []

data = np.empty((len(train_sample['ImageId']),256, 256,3), dtype = np.uint8)
data_target = np.empty((len(train_sample['ImageId'])), dtype = np.uint8)
image_name_list = os.listdir(TRAIN_IMAGES_PATH)
index = 0

for image_name in image_name_list:
    if image_name in list(train_sample['ImageId']):
        imageA = Image.open(os.path.join(TRAIN_IMAGES_PATH, image_name)).resize((256,256)).convert('RGB')
        data[index] = imageA
        data_target[index] = train_sample[train_sample['ImageId'].str.contains(image_name)]['exist_ship'].iloc[0]
        index += 1
        

targets = data_target.reshape(len(data_target), -1)
enc = OneHotEncoder()
enc.fit(targets)
targets = enc.transform(targets).toarray()

x_train, x_val, y_train, y_val = train_test_split(data,targets, test_size = 0.2)

img_gen = ImageDataGenerator(
    rescale=1./255,
    zca_whitening = False,
    rotation_range = 90,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    brightness_range = [0.5, 1.5],
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode='nearest'
    
)

img_width, img_height = 256, 256
model = VGG16Model(weights = 'imagenet', include_top=False, input_shape = (img_width, img_height, 3))

for layer in model.layers:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model_final = Model(inputs = model.input, outputs = predictions)

from keras import optimizers
epochs = 5
lrate = 0.001
adam = optimizers.Adam(lr = lrate, beta_1 = 0.9, beta_2 = 0.999)
model_final.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#model_final.summary()

model_final.fit_generator(img_gen.flow(x_train, y_train, batch_size = 8),steps_per_epoch = len(x_train)/8, validation_data = (x_val,y_val), epochs = epochs )
model_final.save('ResNet_transfer_ship.h5')

train_predict_sample = train_gp.sample(2000)

data_predict = np.empty((len(train_predict_sample['ImageId']),256, 256,3), dtype = np.uint8)
data_target_predict = np.empty((len(train_predict_sample['ImageId'])), dtype = np.uint8)
image_name_list = os.listdir(TRAIN_IMAGES_PATH)
index = 0

for image_name in image_name_list:
    if image_name in list(train_predict_sample['ImageId']):
        imageA = Image.open(os.path.join(TRAIN_IMAGES_PATH, image_name)).resize((256,256)).convert('RGB')
        data_predict[index] = imageA
        data_target_predict[index] = train_predict_sample[train_predict_sample['ImageId'].str.contains(image_name)]['exist_ship'].iloc[0]
        index += 1

targets_predict = data_target_predict.reshape(len(data_target_predict),-1)
enc = OneHotEncoder()
enc.fit(targets_predict)
targets_predict = enc.transform(targets_predict).toarray()

predict_ship = model_final.evaluate(data_predict,targets_predict)

image_test_name_list = os.listdir(TEST_IMAGES_PATH)
data_test = np.empty((len(image_test_name_list), 256, 256,3), dtype=np.uint8)
test_name = []
index = 0

for image_name in image_test_name_list:
    imageA = Image.open(os.path.join(TEST_IMAGES_PATH, image_name)).resize((256,256)).convert('RGB')
    test_name.append(image_name)
    data_test[index]=imageA
    index += 1
print (data_test.shape)

result = model_final.predict(data_test)
result_list = {
    "ImageId": test_name,
    "Have_ship":np.argmax(result,axis=1)
}
result_pd = pd.DataFrame(result_list)
result_pd.to_csv('Have_ship_or_not.csv',index = False)'''