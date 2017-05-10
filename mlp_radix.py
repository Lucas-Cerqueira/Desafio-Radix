# Author: Lucas de Andrade Cerqueira
# Email:  lucas.cerqueira@poli.ufrj.br

import matplotlib.pyplot as plt
import numpy as np
import csv
import keras

from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.cross_validation import train_test_split

num_classes = 2
batch_size = 50
epochs = 400

csvfile = open('p1_data_train.csv', 'r')
reader = csv.reader(csvfile, delimiter=',')
train_data = list(reader)
train_data = np.array(train_data)
train_data = np.delete (train_data, 0, 0).astype('float')   		# Remove first line that contains the headers
csvfile.close()


# Split the array between the inputs and the targets
input_data = train_data[:,0:4]
target_data = train_data[:,4]

input_train, input_test, target_train, target_test = train_test_split (input_data, target_data, test_size=0.25)

# 20 5 1 : 350ep

inputs = Input(shape=(input_train.shape[1],))
hidden = Dense (20, activation="sigmoid") (inputs)
hidden = Dense (10, activation="sigmoid") (hidden)
output = Dense (1, activation="sigmoid") (hidden)

model = Model (inputs=inputs, outputs=output)

model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adagrad(), metrics=['accuracy'])

earlyStop = EarlyStopping(monitor='val_acc', patience=25)

history = model.fit(input_train, target_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(input_test,target_test),
	                callbacks = [earlyStop])


score = model.evaluate(input_test, target_test, verbose=0)
print score

