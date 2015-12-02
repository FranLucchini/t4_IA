import sys
import scipy
import scipy.io
import numpy as np
import posixpath

from pathlib import Path

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

#### atributos y parámatros ###

label = dict()
train_set_path = 'FC6Train'
test_set_path = 'FC6Test'

train_set = dict()
train_set_data = dict()
test_set = dict()

#### main ####
def main(argv = sys.argv):

	print('Train Mode')

	label_vector(0,1,2,3,4,5)
	print_label_vector()
	getTrainInput(0, 1, 2, 3, 4, 5)
	#printTrainSet(10)
	build_neural_net(6)

	label_vector(0,1,10,11,100,101)
	print_label_vector()

	print('Test Mode')

#### Labels ####

def label_vector (v1, v2, v3, v4, v5, v6):
	label['auditorium'] = v1
	label['bar'] = v2
	label['classroom'] = v3
	label['closet'] = v4
	label['movie_theater'] = v5
	label['restaurant'] = v6

def print_label_vector() :
	print(label)

#### Input Vector - Training ####

def getTrainInput(v1, v2, v3, v4, v5, v6):

	getVectorInFolder('AuditoriumTrain', v1)
	getVectorInFolder('barTrain',v2)
	getVectorInFolder('classroomTrain', v3)
	getVectorInFolder('closetTrain', v4)
	getVectorInFolder('movietheaterTrain', v5)
	getVectorInFolder('restaurantTrain', v6)

	
def getVectorInFolder(subfolder, v) :
	p = Path(train_set_path + '/' + subfolder)
	for x in p.iterdir():
		train_set[x] = v
		train_set_data[x] = scipy.io.loadmat(str(x))['stored'][0]


def printTrainSet(lim):
	i = 0;
	for x in train_set:
		print(x)
		print(train_set[x])

		i += 1
		if (i == lim):
			break

def printTrainSetData(lim):
	i = 0
	for x in train_set_data:
		print(x)
		print(train_set_data[x])

		i += 1
		if (i == lim):
			break


#### Neural Net ####

def build_neural_net(num_output) :
	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape:
	# here, 20-dimensional vectors.

	#tenemos una capa oculta para un total de 3 capas.
	#elegimos un 20% de los vectores como set de test
	model.add(Dense(4096, input_dim=4096, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))

	#model.add(Dense(2047, input_dim=4096, init='uniform'))
	#model.add(Activation('tanh'))
	#model.add(Dropout(0.5))

	#model.add(Dense(num_output, input_dim=2047, init='uniform'))
	#model.add(Activation('tanh'))

	#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.compile(loss='mean_squared_error', optimizer='adam')

	X = np.array(list(train_set_data.values()))
	y = list(train_set.values())

	#tokenizer = Tokenizer(nb_words=4096)
	#X_train = tokenizer.sequences_to_matrix(X, mode="binary")

	nb_classes = np.max(y)+1
	y_train = np_utils.to_categorical(y, nb_classes)

	#print(X_train.shape)
	#printElement(X,10)
	#print(y_train.shape)
	#printElement(y_train,10)
	
	print(type(X))
	print(type(y))
	#print(type(y_train))

	hist = model.fit(X, y, nb_epoch=4, batch_size=500)
	print(hist.history)
	#score = model.evaluate(X_test, y_test, batch_size=16)

#### Helpers ####

def printElement(element, lim):
	i = 0
	for elem in element:
		print(elem)
		i += 1
		if (i == lim):
			break

if __name__ == "__main__":
	sys.exit(main())