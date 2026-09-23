import numpy as np
import json

from dataset_reader import DatasetReader
from file_handler import FileHandler
from components.funcs import *
from components.layers import *
from components.processing import pre_process

def load_config_args():
	data = ''
	config: dict = {}
	with open("config.json", 'r') as file:
		data = file.read()
		file.close()
	config = json.loads(data)
	return config

# Loads weights at init, function to save
class Model:
	def __init__(self, settings: dict):
		self.__config = settings
		n_layers = len(self.__config['hidden']) + 1 	# Include output
		self.__weights, self.__biases = FileHandler.load_network(n_layers, n_layers)
		# Extra parameters
		self.__params = FileHandler.load_extra(n_layers - 1)
		self.__n_layers = n_layers

		self.__w1 = Linear(self.__weights[0], self.__biases[0])
		self.__f1 = LeakyReLU(self.__params[0])
		self.__w2 = Linear(self.__weights[1], self.__biases[1])
		self.__f2 = LeakyReLU(self.__params[1])
		self.__w3 = Linear(self.__weights[2], self.__biases[2])

		self.__train_iter = 0
	def __decode(self, layer):
		res = np.argmax(layer.flatten())
		return self.__config['outputs'][res]
	def __encode(self, value):
		# One-hot vector
		size = len(self.__config['outputs'])
		vect = np.zeros(size, dtype=np.float64)
		for res,opt in enumerate(self.__config['outputs']):
			if opt == str(value):
				vect[res] = 1
				return vect.reshape(1,-1)
		raise KeyError("Value not encodable")

	# Standard forward, single-input
	def forward(self, input_layer):
		layers = []
		layer = pre_process(input_layer)
		layer = self.__w1.forward(layer)
		layer = self.__f1.forward(layer)
		layers.append(np.copy(layer))
		layer = self.__w2.forward(layer)
		layer = self.__f2.forward(layer)
		layers.append(np.copy(layer))
		layer = self.__w3.forward(layer)
		layer = Softmax.forward(layer)
		return layer, layers

	# Forward used for analysis purposes, takes batch
	# (entire dataset usually)
	def forward_analysis(self, batch_input, expected_output):
		out = np.vstack([self.__encode(o) for o in expected_output])
		layer, _ = self.forward(batch_input)
		# Calculate loss
		loss = cross_entropy_loss(layer, out)
		# Calculate accuracy
		cmps = np.argmax(out, axis=1) == np.argmax(layer, axis=1)
		return loss, np.count_nonzero(cmps)/len(cmps)

	# Backprop, will perform forward by itself, tracking layers
	def backprop(self, batch_input, expected_output):
		out = np.vstack([self.__encode(o) for o in expected_output])
		batch_size = self.__config['batch_size']
		# Standard forward
		layer, layers = self.forward(batch_input)
		# Backprop, grads calculated and weights updated
		deriv = cost_deriv(layer, out)
		deriv = self.__w3.back(deriv)
		deriv = self.__f1.back(layers[-1], deriv)
		deriv = self.__w2.back(deriv)
		deriv = self.__f2.back(layers[-2], deriv)
		deriv = self.__w1.back(deriv)

	# Loop through each back
	def train(self, images, labels):
		for b_img, b_lbl in zip(images, labels):
			self.backprop(b_img, b_lbl)

	# Network saver
	def save(self):
		FileHandler.save_network(self.__weights, self.__biases)
		if self.__params is None:
			return
		FileHandler.save_extra(self.__params)
