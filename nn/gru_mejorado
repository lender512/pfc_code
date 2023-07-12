from sklearn.model_selection import validation_curve
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
import os
import math
from keras.callbacks import LearningRateScheduler
from keras.layers import LSTM
import ast
import h5py
import pandas as pd
from sklearn.metrics import roc_curve

train_data = np.array([])
train_labels = np.array([])
val0_data = np.array([])
val0_labels = np.array([])
val1_data = np.array([])
val1_labels = np.array([])
input_shape = (30, 6)

with h5py.File('nn/Rand_WS30.h5', 'r') as hdf:
    data = hdf.get('train_data')
    train_data = np.array(data)
    data = hdf.get('train_labels')
    train_labels = np.array(data)

# get the data of label 0
for i in range(1000):
    if train_labels[i] == 0:
        val0_data = np.append(val0_data, train_data[i])
# create a labels in the amount of data
val0_labels = np.empty(int(val0_data.shape[0] / (input_shape[0] * 6)))
val0_labels.fill(0)
# get the data of label 1
for i in range(1000):
    if train_labels[i] == 1:
        val1_data = np.append(val1_data, train_data[i])
# create a labels in the amount of data
val1_labels = np.empty(int(val1_data.shape[0] / (input_shape[0] * 6)))
val1_labels.fill(1)

# Reshape the data to an array of windows of size ws
val1_data = val1_data.reshape(int(val1_labels.shape[0]), input_shape[0], 6)
val0_data = val0_data.reshape(int(val0_labels.shape[0]), input_shape[0], 6)

# remove test data from the whole data
train_data = train_data[1000:]
train_labels = train_labels[1000:]
model = keras.models.Sequential()
model.add(layers.Conv1D(32, 2, activation='relu', input_shape=input_shape))
model.add(layers.GRU(32, return_sequences=True))
model.add(layers.Dropout(0.5))
model.add(layers.GRU(32, return_sequences=True))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))  # Changed the number of neurons to 1 and activation to sigmoid
print(model.summary())

loss = tf.keras.losses.BinaryCrossentropy()
optim = keras.optimizers.Adam(learning_rate=0.0)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)
batchSize = 32
epochs = 150


def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.1
    epochs_drop = 100.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
with tf.device('/cpu:0'):
    model.fit(train_data, train_labels, batch_size=batchSize, validation_split=0.1, callbacks=callbacks_list,
              epochs=epochs, shuffle=True)
    
val0_data.shape
val0_labels.shape

model.evaluate(val0_data,val0_labels,batch_size=batchSize, verbose = 1)

probability_model = keras.models.Sequential([
    model,
])

predictions1 = probability_model.predict(val0_data)

pre0 = predictions1[0]

print(pre0)

model.evaluate(val1_data,val1_labels,batch_size=batchSize, verbose = 1)

probability_model = keras.models.Sequential([
    model,
])

predictions2 = probability_model.predict(val1_data)

pre1 = predictions2[0]

print(pre1)




df = pd.DataFrame()

for i in range(len(predictions1)):
    df = df._append({'Prediction': predictions1[i][0], 'Label': 0}, ignore_index=True)
    #use concat instead of append


for i in range(len(predictions2)):
    df = df._append({'Prediction': predictions2[i][0], 'Label': 1}, ignore_index=True)

df

predictions = df['Prediction']
labels = df['Label']

# Calculate the false positive rate (FPR) and true positive rate (TPR) for various thresholds
fpr, tpr, thresholds = roc_curve(labels, predictions)

# Calculate the false negative rate (FNR) as 1 - TPR
fnr = 1 - tpr

# Find the threshold index that results in the closest FPR and FNR
eer_threshold_idx = np.argmin(np.abs(fpr - fnr))

# Calculate the Equal Error Rate (EER) using the average of closest FPR and FNR
eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2

print("Equal Error Rate (EER):", eer)
# Equal Error Rate (EER): 0.048998195992783995

model.save("nn/2GRU.h5")
