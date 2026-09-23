import numpy as np
import json
import random

NETWORK_PATH = "network_data/"
WEIGHT_FILES = lambda c: NETWORK_PATH + f"w{c}"
BIAS_FILES = lambda c: NETWORK_PATH + f"b{c}"
FUNCPARAM_FILES = lambda c: NETWORK_PATH + f"f{c}"



def get_network_specs():
	specs = []
	config = {}
	data = ""
	with open("config.json", 'r') as file:
		data = file.read()
		file.close()
	config = json.loads(data)
	specs.append(config['inputs'])
	specs += config['hidden']
	specs.append(len(config['outputs']))
	return specs



# Layer constructions

# Weight: He init
# Bias: 0 bias
# Extra: Depends on layer type

def construct_weights(size):
	weights = []
	for prev_layer, next_layer in zip(size[:-1], size[1:]):
		layer_weights = np.random.randn(prev_layer, next_layer) * np.sqrt(2 / prev_layer)
		weights.append(layer_weights.reshape((prev_layer, next_layer)))
	return weights

def construct_biases(size):
	biases = []
	for prev_layer, next_layer in zip(size[:-1], size[1:]):
		layer_biases = np.zeros(next_layer)
		biases.append(layer_biases.reshape(next_layer, 1))
	return biases


def construct_funcparams(size, count, modifier):
	func_params = []
	for hidden_layer in size[1:-1]:
		layer_func_params = np.ones((hidden_layer,count))
		layer_func_params *= 1/100
		func_params.append(layer_func_params)
	return func_params

def construct_network(size, extra):
	weights = construct_weights(size)
	biases = construct_biases(size)
	if extra["use_params"]:
		params = construct_funcparams(size, extra["count"], extra["modifier"])
		return (weights, biases, params)
	return (weights, biases)

# Layer saving
def save_weights(weights):
	for w, weight in enumerate(weights):
		np.savetxt(WEIGHT_FILES(w), weight)

def save_biases(biases):
	for b, bias in enumerate(biases):
		np.savetxt(BIAS_FILES(b), bias)

def save_funcs(params):
	for p, param in enumerate(params):
		np.savetxt(FUNCPARAM_FILES(p), param)

def save_network(network):
	save_weights(network[0])
	save_biases(network[1])
	if len(network) == 3:
		save_funcs(network[2])


# Create a network and save it
def create_network(extra):
	size = get_network_specs()
	network = construct_network(size, extra)
	save_network(network)