#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Author: Lucas de Andrade Cerqueira
# Email: lucas.cerqueira@poli.ufrj.br

import os.path
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import matplotlib.pyplot as plt
import numpy as np
import csv
import itertools
from sklearn.externals import joblib

plt.close('all')
num_classes = 2

def plot_temp(temp_y, temp_x, axis=None, show=False):
	if (temp_y < 1) or (temp_y > 4) or (temp_x < 1) or (temp_x > 4):
		return

	if (not axis):
		plt.figure()
		plt.plot (input_data[target_data == 0,temp_x-1], input_data[target_data == 0,temp_y-1], 'bo', label='Estado ótimo')
		plt.plot (input_data[target_data == 1,temp_x-1], input_data[target_data == 1,temp_y-1], 'r*', label='Estado regular')
		plt.xlabel ('Temperatura '+str(temp_x)+' (°C)', fontsize=14)
		plt.ylabel ('Temperatura '+str(temp_y)+' (°C)', fontsize=14)
		plt.title ('Temp'+str(temp_y)+' x Temp'+str(temp_x)+' por classe', fontsize="x-large", fontweight='bold')
		plt.legend()
		if (show):
			plt.show()
	else:
		axis.plot (input_data[target_data == 0,temp_x-1], input_data[target_data == 0,temp_y-1], 'bo', label='Estado ótimo')
		axis.plot (input_data[target_data == 1,temp_x-1], input_data[target_data == 1,temp_y-1], 'r*', label='Estado regular')
		axis.set_xlabel ('Temperatura '+str(temp_x)+' (°C)', fontsize=14)
		axis.set_ylabel ('Temperatura '+str(temp_y)+' (°C)', fontsize=14)
		axis.set_title ('Temp'+str(temp_y)+' x Temp'+str(temp_x)+' por classe', fontsize="x-large", fontweight='bold')
		axis.legend()



csvfile = open('data/p1_data_train.csv', 'r')
reader = csv.reader(csvfile, delimiter=',')
train_data = list(reader)
train_data = np.array(train_data)
train_data = np.delete (train_data, 0, 0).astype('float')   		# Remove first line that contains the headers
csvfile.close()

# Split the array between the inputs and the targets
input_data = train_data[:,0:4]
target_data = train_data[:,4]


# Plot histogram of data for each class
fig, axarr = plt.subplots(2, sharex=True)
plt.set_cmap('jet_r')

for i in range(num_classes):
	array = input_data[target_data == i]
	
	axarr[i].hist(array, bins='auto', label=['Temp. 1', 'Temp. 2', 'Temp. 3', 'Temp. 4'])
	axarr[i].legend()
	axarr[i].set_ylabel('Nº de ocorrências', fontsize=14)
	axarr[i].yaxis.set_ticks(np.arange(0, 500, 50))
	
	if (i==0):
		axarr[i].set_title('Estado ótimo', fontweight='bold')
		
	else:
		axarr[i].set_title('Estado regular', fontweight='bold')


axarr[1].set_xlabel('Temperatura (°C)', fontsize=14)
fig.suptitle ('Histograma das temperaturas por estado no conjunto de treinamento', fontsize="x-large", fontweight='bold')
plt.savefig ('pictures/histograma_por_estado.pdf', bbox_inches='tight')


# Plot Temp4 x Temp2 for each class
plot_temp(4,2)
plt.savefig ('pictures/temp4xtemp2.pdf')



# Plot Temp3 x Temp1 for each class
plot_temp (3,1)
plt.savefig ('pictures/temp3xtemp1.pdf')


# Load model to plot confusion matrix if one has been trained
if (not os.path.exists('model.hdf5')):
  print ('Model file not found')
  quit()

if (not os.path.exists('scaler.save')):
	print ('Scaler data file not found')
	quit()

from keras.models import load_model
from sklearn.metrics import confusion_matrix

model = load_model('model.hdf5')
scaler = joblib.load ('scaler.save')

predicted = model.predict(scaler.transform(input_data))
predicted = (predicted[:,0]>0.5).astype('int')

# Modified Scikit learn example of confusion matrix plot
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, title='Matriz de confusão'):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greys)
    plt.title(title, fontweight = 'bold', fontsize='x-large')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    thresh = 50
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "%.2f" % cm[i, j] +"%",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('Alvo', fontsize = 14)
    plt.xlabel('Saída', fontsize = 14)



# Compute confusion matrix
cnf_matrix = (confusion_matrix(target_data, predicted)).astype('float')
cnf_matrix[0,:] = cnf_matrix[0,:]*100./sum(target_data==0) 
cnf_matrix[1,:] = cnf_matrix[1,:]*100./sum(target_data==1)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Estado ótimo', 'Estado regular'])
plt.savefig ('pictures/confusion_matrix.pdf', bbox_inches='tight')