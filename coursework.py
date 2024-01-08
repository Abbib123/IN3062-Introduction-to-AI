# -*- coding: utf-8 -*-

import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.utils
import seaborn as sns
from tensorflow import keras
from sklearn import metrics
from PIL import Image
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

path = "."  

filename_read = os.path.join(path, "full_df.csv")
df = pd.read_csv(filename_read)
# print(df[0:6392])

df = pd.read_csv(filename_read, na_values=['NA', '?'])

headers = list(df.columns.values)
fields = []

for field in fields:
    print(field)

# Shows before an after drop of columns to show what we had and what we are using for the model
# print(f"before drop: {df.columns}")
df.drop('ID',axis=1, inplace=True)
df.drop('N',axis=1, inplace=True)
df.drop('D',axis=1, inplace=True)
df.drop('G',axis=1, inplace=True)
df.drop('C',axis=1, inplace=True)
df.drop('A',axis=1, inplace=True)
df.drop('H',axis=1, inplace=True)
df.drop('M',axis=1, inplace=True)
df.drop('O',axis=1, inplace=True)
df.drop('Left-Fundus',axis=1, inplace=True)
df.drop('Right-Fundus',axis=1, inplace=True)
df.drop('Left-Diagnostic Keywords',axis=1, inplace=True)
df.drop('Right-Diagnostic Keywords',axis=1, inplace=True)
df.drop('filepath',axis=1, inplace=True)
df.drop('target',axis=1, inplace=True)
df.drop('Patient Age', axis=1,inplace=True)
df.drop('Patient Sex', axis=1,inplace=True)
# print(f"after drop: {df.columns}")
# print(df[0:6392])
# print(df.head())

width = 128 
height = 128
def preprocess_image(file_paths):
    images = []
    for file_path in file_paths:
        full_path = './images/' + file_path
        # Load and preprocess image
        img = Image.open(full_path)
        img = img.resize((width, height))
        img_array = np.array(img) / 255.0
        images.append(img_array)
    return np.array(images)

# Code for Splitting preprocessed data:
# TOTAL = 6391 separate rows of data we want approx a 3:1:1 split Training:Validation:Test
# Training = 3835 out of 6391 rows
# Validation = 1278 out of 6391 rows
# Test = 1278 out of 6391 rows

# Prepping test train split for x and y
X = df.filename
y = df.labels
X_train_processed = preprocess_image(X)
#y = tf.keras.utils.to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X_train_processed,y,test_size=0.2)

# Expand to 3 dimensional
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# result = (X_train_processed[:1])
np.set_printoptions(threshold=np.inf)
# print(result)

# SMOTE Code for balancing data (NEED Data splitting first)
# Code provided by https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/
#smote_count = Counter(y_train)
#print ('Before', smote_count)# Show the count before smote
#smt = SMOTE()
#X_train_smt, y_train_smt = smt.fit_resample(X_train_processed, y_train)
#post_smote_count = Counter(y_train_smt)
#print('After', post_smote_count)

# Flatten data to fit for smote

# X_train_processed_flat = X_train_processed.reshape(X_train_processed.shape[0], -1)
# print("Shape before SMOTE:", X_train_processed_flat.shape)
# smt = SMOTE()
# X_train_smt, y_train_smt = smt.fit_resample(X_train_processed, y_train)
# post_smote_count = Counter(y_train_smt)
# print('After', post_smote_count)

#testing scikit learn



#4. preparing to build the network

batch_size = 128
print(y_train.shape)
print("Y shape^")
num_classes = len(np.unique(y_train.shape[0]))
epochs = 32
save_dir = './' 
model_name = 'keras_lfw_trained_model.h5'

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', input_shape= (128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

#X_train = np.squeeze(X_train, axis=3)
history = model.fit(X_train,y_train,verbose=1,epochs=24)

#5. make predictions

#make predictions (will give a probability distribution)

pred = model.predict(X_test)
print(pred)
pred = np.argmax(pred,axis=1)
print(pred)
print("pred shape^")
print(y_test.shape)
print("y test shape^")
y_compare = np.argmax(y_test)
print(y_compare)
#and calculate accuracy
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))

#6. plot data 

# Plot training & validation loss values

print(history.history.keys())
plt.plot(history.history['loss'])
plt.title('Model loss/accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper left')

plt=plt.twinx()
color = 'red'
plt.plot(history.history['accuracy'],color=color)
plt.ylabel('Accuracy')
plt.legend(['Accuracy'], loc='upper center')
plt.show()

#7. add confusion matrix to testing data

# look layer by layer using activation maps for model analysis 

#recommended = CNN's | SVM's | KNN's | accuracy matrix

# model2 = Sequential()
# model2.add(Conv2D(64, kernel_size=(4, 4), activation='relu', strides=1, padding='same', input_shape= X_train[0].shape))
# model2.add(Conv2D(32, (3, 3), activation='relu'))
# model2.add(MaxPooling2D(pool_size=(2, 2)))
# model2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model2.add(Conv2D(64, (3, 3), activation='relu'))
# model2.add(MaxPooling2D(pool_size=(2, 2)))

# model2.add(Flatten())
# model2.add(Dense(128, activation='relu'))

# model2.add(Dense(num_classes))
# model2.add(Activation('softmax'))