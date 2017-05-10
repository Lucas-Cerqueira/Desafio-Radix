#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Author: Lucas de Andrade Cerqueira
# Email: lucas.cerqueira@poli.ufrj.br

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import matplotlib.pyplot as plt
import numpy as np
import csv

plt.close('all')
num_classes = 2

def plot_temp(temp_y, temp_x, axis=None):
	if (temp_y < 1) or (temp_y > 4) or (temp_x < 1) or (temp_x > 4):
		return

	if (not axis):
		plt.figure()
		plt.plot (train_data[train_data[:,4] == 0,temp_x-1], train_data[train_data[:,4] == 0,temp_y-1], 'r*', label='Estado regular')
		plt.plot (train_data[train_data[:,4] == 1,temp_x-1], train_data[train_data[:,4] == 1,temp_y-1], 'bo', label='Estado ótimo')
		plt.xlabel ('Temperatura '+str(temp_x)+' (°C)', fontsize=14)
		plt.ylabel ('Temperatura '+str(temp_y)+' (°C)', fontsize=14)
		plt.title ('Temp'+str(temp_y)+' x Temp'+str(temp_x)+' por classe', fontsize="x-large", fontweight='bold')
		plt.legend()
		plt.show()
	else:
		print ("Entrou")
		axis.plot (train_data[train_data[:,4] == 0,temp_x-1], train_data[train_data[:,4] == 0,temp_y-1], 'r*', label='Estado regular')
		axis.plot (train_data[train_data[:,4] == 1,temp_x-1], train_data[train_data[:,4] == 1,temp_y-1], 'bo', label='Estado ótimo')
		axis.set_xlabel ('Temperatura '+str(temp_x)+' (°C)', fontsize=14)
		axis.set_ylabel ('Temperatura '+str(temp_y)+' (°C)', fontsize=14)
		axis.set_title ('Temp'+str(temp_y)+' x Temp'+str(temp_x)+' por classe', fontsize="x-large", fontweight='bold')
		axis.legend()



csvfile = open('p1_data_train.csv', 'r')
reader = csv.reader(csvfile, delimiter=',')
train_data = list(reader)
train_data = np.array(train_data)
train_data = np.delete (train_data, 0, 0).astype('float')   		# Remove first line that contains the headers


# Plot histogram of data for each class
fig, axarr = plt.subplots(2, sharex=True)
plt.set_cmap('jet_r')

for i in range(num_classes):
	indexes = (train_data[:,4] == i)
	array = train_data[indexes]
	array = array[:,0:4]
	
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
plt.show()


# fig, axarr = plt.subplots(1,3)

# for i in range(axarr.shape[0]):
# 	plot_temp (i+2, 1, axarr[i])	

# plt.show()


# fig, axarr = plt.subplots(1,2)

# for i in range(axarr.shape[0]):
# 	plot_temp (i+3, 2, axarr[i])	

# plt.show()

# plot_temp (4,3)

#Plot Temp4 x Temp2 for each class
plot_temp(4,2)

#Plot Temp3 x Temp1 for each class
plot_temp (3,1)

