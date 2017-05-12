#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Author: Lucas de Andrade Cerqueira
# Email:  lucas.cerqueira@poli.ufrj.br

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import itertools
import matplotlib.pyplot as plt
import numpy as np
import csv
import keras

from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


num_classes = 2
batch_size = 50
epochs = 300

csvfile = open('data/p1_data_train.csv', 'r')
reader = csv.reader(csvfile, delimiter=',')
train_data = list(reader)
train_data = np.array(train_data)
train_data = np.delete (train_data, 0, 0).astype('float')   		# Remove first line that contains the headers
csvfile.close()

# Split the array between the inputs and the targets
input_data = train_data[:,0:4]
target_data = train_data[:,4]


# Split input_data between training and test set (0.75, 0.25)
input_train, input_test, target_train, target_test = train_test_split (input_data, target_data, test_size=0.25)


# Normalize the input and save the scaler fit
scaler = StandardScaler().fit(input_train)
input_train = scaler.transform (input_train)
input_test = scaler.transform (input_test)
joblib.dump (scaler, 'scaler.save')


# Create the network structure
inputs = Input(shape=(input_train.shape[1],))
hidden = Dense (20, activation="sigmoid") (inputs)
hidden = Dense (10, activation="sigmoid") (hidden)
output = Dense (1, activation="sigmoid") (hidden)

model = Model (inputs=inputs, outputs=output)

# Compile the model using MSE as loss function and optimizer Adagrad
model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6), metrics=['accuracy'])


# Configure callbacks to be used during the training
earlyStop = EarlyStopping (monitor='val_acc', patience=50)
checkpoint = ModelCheckpoint ('model.hdf5', monitor='val_acc', save_best_only=True)

# Train the model and save the best using the "ModelCheckpoint" callback
history = model.fit(input_train, target_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(input_test,target_test),
	                callbacks = [earlyStop, checkpoint])

# Evaluate the model using test set
model = load_model('model.hdf5')
score = model.evaluate(input_test, target_test, verbose=0)
print ("Accuracy on test set: "+ str(score[1]))


fig, ax = plt.subplots(1,2)

ax[0].plot (history.history['val_loss'], 'r', linewidth=2)
ax[0].set_title ('MSE no conjunto de teste por época', fontsize='x-large', fontweight='bold')
ax[0].set_xlabel('#Época', fontsize=14, fontweight = 'bold')
ax[0].set_ylabel ('MSE', fontsize=14, fontweight = 'bold')

ax[1].plot (history.history['val_acc'], 'r', linewidth=2)
ax[1].set_title ('Acurácia no conjunto de teste por época', fontsize='x-large', fontweight='bold')
ax[1].set_xlabel('#Época', fontsize=14, fontweight = 'bold')
ax[1].set_ylabel ('Acurácia', fontsize=14, fontweight = 'bold')

best_epoch = (np.argmax(history.history['val_acc']))

ax[1].hold(True)
ax[1].plot (best_epoch, history.history['val_acc'][best_epoch], 'o', color='black')
ax[1].annotate('%.3f'%history.history['val_acc'][best_epoch], xy=(best_epoch, history.history['val_acc'][best_epoch]), xytext=(best_epoch, history.history['val_acc'][best_epoch]-0.015))
#fig.savefig('pictures/loss_acc_epoch.svg', bbox_inches='tight')
plt.show()