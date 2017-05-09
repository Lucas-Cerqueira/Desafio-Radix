# Author: Lucas de Andrade Cerqueira
# Email: lucas.cerqueira@poli.ufrj.br

import matplotlib.pyplot as plt
import numpy as np
import csv
import keras

from keras.layers import Input, Dense
from keras.models import Model

num_classes = 2
batch_size = 100
epochs = 100

csvfile = open('p1_data_train.csv', 'r')
reader = csv.reader(csvfile, delimiter=',')
train_data = list(reader)
train_data = np.array(train_data)
train_data = np.delete (train_data, 0, 0).astype('float')   		# Remove first line that contains the headers


# Split the array between the inputs and the targets
input_data = train_data[:,0:4]
target_data = train_data[:,4]

inputs = Input(shape=(input_data.shape[1],))

hidden = Dense (10, activation="sigmoid") (inputs)
hidden = Dense (4, activation="sigmoid") (hidden)
output = Dense (1, activation="sigmoid") (hidden)

model = Model (inputs=inputs, outputs=output)
model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(input_data, target_data, batch_size=batch_size, epochs=epochs, verbose=1)
score = model.evaluate(input_data, target_data, verbose=0)

