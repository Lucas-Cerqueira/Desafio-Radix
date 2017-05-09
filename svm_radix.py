# Author: Lucas de Andrade Cerqueira
# Email: lucas.cerqueira@poli.ufrj.br

from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import csv

plt.close('all')

num_classes = 2

csvfile = open('p1_data_train.csv', 'r')
reader = csv.reader(csvfile, delimiter=',')
train_data = list(reader)
train_data = np.array(train_data)
train_data = np.delete (train_data, 0, 0).astype('float')   		# Remove first line that contains the headers

indexes = (train_data[:,4] == 1)

input_data = train_data[:,0:3]
target_data = train_data[:,4]

model = svm.SVC(gamma=0.001, C=100., cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, kernel='rbf', verbose=True)
model.fit (input_data, target_data)
score = model.score (input_data, target_data)

print (score)