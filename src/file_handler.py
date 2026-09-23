import numpy as np
import os

C_DIR = os.path.dirname(os.path.abspath(__file__))

NETWORK_PATH = C_DIR + "/network_data/"
BACKUP_PATH = C_DIR + "/network_backup/"

WEIGHT_FILE = lambda n: NETWORK_PATH + f"w{n}"
BIAS_FILE = lambda n: NETWORK_PATH + f"b{n}"
PARAM_FILE = lambda n: NETWORK_PATH + f"f{n}"

class FileHandler:
	# Backup handling for network weights(saving/loading)
	def create_backup():
		files = os.listdir(NETWORK_PATH)
		for f in files:
			if f == "param_display.py":
				continue
			content = b""
			with open(NETWORK_PATH + f, "rb") as file:
				content = file.read()
				file.close()
			with open(BACKUP_PATH + f, "wb") as file:
				file.write(content)
				file.close()
	def load_backup():
		files = os.listdir(BACKUP_PATH)
		for f in files:
			content = b""
			with open(BACKUP_PATH + f, "rb") as file:
				content = file.read()
				file.close()
			with open(NETWORK_PATH + f, "wb") as file:
				file.write(content)
				file.close()

	# Network loading - only weight and bias (linear)
	def load_network(weight_spec, bias_spec):
		weights = []
		biases = []
		for w in range(weight_spec):
			weight = np.loadtxt(WEIGHT_FILE(w), dtype="float64")
			weights.append(weight)
		for b in range(bias_spec):
			bias = np.loadtxt(BIAS_FILE(b), dtype="float64")
			biases.append(bias)
		return (weights, biases)

	# Extra parameter handling for functions
	def load_extra(params_spec):
		params = []
		for p in range(params_spec):
			param = np.loadtxt(PARAM_FILE(p), dtype="float64")
			params.append(param)
		return params
	def save_network(weights, biases):
		for w, weight in enumerate(weights):
			np.savetxt(WEIGHT_FILE(w), weight)
		for b, bias in enumerate(biases):
			np.savetxt(BIAS_FILE(b), bias)
	def save_extra(params):
		for p, param in enumerate(params):
			np.savetxt(PARAM_FILE(p), param)