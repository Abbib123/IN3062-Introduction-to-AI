# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix

path = "."  

filename_read = os.path.join(path, "full_df.csv")
df = pd.read_csv(filename_read)
print(df[0:6392])

df = pd.read_csv(filename_read, na_values=['NA', '?'])

headers = list(df.columns.values)
fields = []

for field in fields:
    print(field)

# Shows before an after drop of columns to show what we had and what we are using for the model
print(f"before drop: {df.columns}")
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
print(f"after drop: {df.columns}")
print(df[0:6392])
print(df.head())

width = 128 
height = 128
def preprocess_image(file_paths):
    images = []
    for file_path in file_paths:
        # Load image
        img = Image.open(file_path)
        
        # Preprocess: Resize, convert to array, normalize
        img = img.resize((width, height))
        img_array = np.array(img) / 255.0  # Normalize pixel values
        
        images.append(img_array)
    
    return np.array(images)

# Code for Splitting preprocessed data
# TOTAL = 6391 separate rows of data we want approx a 3:1:1 split Training:Validation:Test
# Training = 3835 out of 6391 rows
# Validation = 1278 out of 6391 rows
# Test = 1278 out of 6391 rows
X = df.filename
y = df.labels

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train_processed = preprocess_image(X_train)

# SMOTE Code for balancing data (NEED Data splitting first)
# Code provided by https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/
smote_count = Counter(y_train)
print ('Before', smote_count)# Show the count before smote
smt = SMOTE()
X_train_smt, y_train_smt = smt.fit_resample(X_train_processed, y_train)
post_smote_count = Counter(y_train_smt)
print('After', post_smote_count)

# FOR CNN STRUCTURE (DRAFT):
# For model, Sequential()
# 32 filters, Kernel size of 5x5 , 1 stride, Input shape?? Not sure yet, Same Padding
# RELU activation
# Pooling of 2,2
# Flatten
# Number of layers? Unsure yet but want maybe 3-4?
# Continue layering, pooling, flattening etc
# Condense to the number of labels (8 if we can get all working but limit to 2 at the beginning for a binary relation)
# Softmax to determine % chance of which it will be 
# Plot results on a graph with pyplot and determine what we get