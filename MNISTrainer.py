import tflearn
import numpy as np
import tflearn.datasets.mnist as mnist
import pickle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


testX = testX.reshape([-1, 28, 28, 1])
data = np.array([mpimg.imread(name) for file in os.listdir('fs/A/a')], dtype=np.float64)
network = input_data(shape=[None, 106, 109, 3], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
loss = 'categorical_crossentropy', name='target')

model = tflearn.DNN(network=network)
model.fit(X_inputs=X, Y_targets=Y,n_epoch=9, validation_set=({'input':testX}, {'target':testY}))
print model.evaluate(testX, testY)[0]
model.save('mnist_model.tfl')

