#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Author: Lucas de Andrade Cerqueira
# Email:  lucas.cerqueira@poli.ufrj.br

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import matplotlib.pyplot as plt
import numpy as np
import csv
import keras
import os.path

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

if (not os.path.exists('scaler.save')):
  print ('Scaler data file not found')
  quit()


csvfile = open('data/p1_data_test.csv', 'r')
reader = csv.reader(csvfile, delimiter=',')
test_data = list(reader)
test_data = np.array(test_data)
test_data = np.delete (test_data, 0, 0).astype('float')   		# Remove first line that contains the headers
csvfile.close()


# Normalize the input
scaler = joblib.load ('scaler.save')
test_data = scaler.transform (test_data)


# Load the trained model
if (not os.path.exists('model.hdf5')):
  print ('Model file not found')
  quit()
model = load_model('model.hdf5')

# Compile the model using MSE as loss function and optimizer Adagrad
#model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adagrad(), metrics=['accuracy'])

# Predict the output for the test data
predicted = model.predict (test_data)


# Make the output either 0 or 1 based on the prediction
predicted = (predicted[:,0]>0.5).astype('int')
predicted = predicted.tolist()

# Write to the file according to the given template
resultFile = open ("p1_predictions.csv", "wb")
wr = csv.writer (resultFile, dialect='excel', delimiter =',', lineterminator='\r\n', quotechar="'")
wr.writerow(['\"target\"'])
for element in predicted:
  wr.writerow(['\"' + str(element) + '\"'])
resultFile.close()

quit()
