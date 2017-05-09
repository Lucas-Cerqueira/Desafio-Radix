# Author: Lucas de Andrade Cerqueira
# Email: lucas.cerqueira@poli.ufrj.br



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

print (indexes)

print (train_data[indexes])


plt.plot (train_data[indexes,0], train_data[indexes,1])

# fig, axarr = plt.subplots(2, sharex=True)

# for i in range(num_classes):
	# indexes = (train_data[:,4] == i)
	# array = train_data[indexes]
	
	# axarr[i].hist(array, bins='auto', label=['Temp. 1', 'Temp. 2', 'Temp. 3', 'Temp. 4'])
	# axarr[i].legend()
	# axarr[i].set_ylabel('Nº de ocorrências', fontsize=14)
	
	# if (i==0):
		# axarr[i].set_title('Estado ótimo', fontweight='bold')
		
	# else:
		# axarr[i].set_title('Estado regular', fontweight='bold')


# axarr[1].set_xlabel('Temperatura (°C)', fontsize=14)
# fig.suptitle ('Histograma das temperaturas por estado no conjunto de treinamento', fontsize="x-large", fontweight='bold')
# plt.show()

