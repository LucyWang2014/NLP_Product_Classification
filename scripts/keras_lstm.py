from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

import nordstrom

def train_lstm(
	max_features = 
	output_dim = 128,
	maxlen = None,
	activation = 'sigmoid'
	inner_activation = 'hard_sigmoid'
	dropout =  0.5,
	data = None,
	batch_size = 16,
	nb_epoch = 10,
	optimizer = 'adadelta',
	loss = 'categorical_crossentropy',
	verbose = 2,
	show_accuracy = True,
	validation_split = 0.1
	):
	
	train,

	model = Sequential()
	model.add(Embedding(max_features, 256, input_length=maxlen))
	model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
	model.add(Dropout(dropout))
	model.add(Dense(1))
	model.add(Activation(activation))

	model.compile(loss=loss, optimizer=optimizer)

	model.fit(X_train, Y_train, batch_size=batch_size, validation_split=validation_split,
		nb_epoch=nb_epoch, verbose=verbose, show_accuracy=show_accuracy)
	score = model.evaluate(X_test, Y_test, batch_size=batch_size)







