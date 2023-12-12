# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

import os
import pandas as pd

path = "."  

filename_read = os.path.join(path, "full_df.csv")
df = pd.read_csv(filename_read)
print(df[0:6392])



