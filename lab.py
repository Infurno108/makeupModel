from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from keras.models import Sequential, Model
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import kagglehub
import warnings
from PIL import Image
from tensorflow.keras.utils import load_img
#warnings.filterwarnings("ignore")

BASE_DIR = '/Users/flintrose/.cache/kagglehub/datasets/jangedoo/utkface-new/versions/1/utkface_aligned_cropped/UTKFace'

image_paths = []
age_labels = []
gender_labels = []

for filename in (os.listdir(BASE_DIR)):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')

    # age from 0
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)

# Convert to DataFrame
df = pd.DataFrame()
df['image_path'], df['age'], df['gender'] = image_paths, age_labels, gender_labels


# print(df.head())
img = Image.open(df['image_path'][2])


# plt.imshow(img)

# sns.histplot(df['age'], kde=True)
# plt.figure(figsize=(20, 20))
files = df.iloc[0:25]

gender_dict = {0: "Male", 1: "Female"}  # Assuming gender dictionary is defined

# for index, row in files.iterrows():
#    plt.subplot(5, 5, index + 1)
#    img = load_img(row['image_path'])
#    img = np.array(img)
#    plt.imshow(img)
#    plt.title(f"Gender: {gender_dict[row['gender']]}, Age: {row['age']}")
#    plt.axis('off')


def extract_features(images):
    features = []
    for image in images:
        # Updated for TensorFlow's load_img
        img = load_img(image, color_mode='grayscale')
        # Updated for PIL's resize
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    features = features.astype('float32') / 255.0  # Normalize pixel values
    return features


features = extract_features(df['image_path'])
print(features.shape)

y_gender = np.array(df['gender'])
y_age = np.array(df['age'])

print(y_age)

# plt.show()


# now we actually build the model ^.^
input_shape = (128, 128, 1)
# input shape declartaion
inputs = Input(shape=input_shape)

# convolutional layers 1
conv_1 = Conv2D(32, kernal_size=(3, 3), activation='relu')(inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

# convolutional layers 2
conv_2 = Conv2D(32, kernal_size=(3, 3), activation='relu')(maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

# convolutional layers 3
conv_3 = Conv2D(32, kernal_size=(3, 3), activation='relu')(maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

# Convolutional layers 4
conv_4 = Conv2D(32, kernal_size=(3, 3), activation='relu')(maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(maxp_4)

# connect layers

dense_1 = Dense(256, activation='relu')(flatten)
dense_2 = Dense(256, activation='relu')(flatten)

# droput layers
dropout_1 = Dropout(0.3)(dense_1)
dropout_2 = Dropout(0.3)(dense_2)

# output layers
output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
output_2 = Dense(1, activation='relu', name='age_out')(dropout_2)

# model
model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'mae'],
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x=X, y=[y_gender, y_age],
                    batch_size=32, epochs=30, validation_split=0.2)

model.save('model.h5')
