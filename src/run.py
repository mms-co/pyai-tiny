import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from components.network_init import create_network

from model import Model
from dataset_reader import DatasetReader
from file_handler import FileHandler


C_DIR = os.path.dirname(os.path.abspath(__file__))

def get_config():
	content = ''
	with open('config.json', 'r') as file:
		content = file.read()
	settings = json.loads(content)
	return settings




def init_command():
	extra_info = {
		"use_params": True,
		"count": 1,
		"modifier": 1/100
	}
	create_network(extra_info)

def train_command():
	config = get_config()
	batch_size = config["batch_size"]
	dataset = DatasetReader(config["dataset"])
	training_set = dataset.load_training()
	train_img = dataset.generate_batches(training_set["images"], batch_size)
	train_label = dataset.generate_batches(training_set["labels"], batch_size)
	model = Model(config)
	epochs = config["epochs"]
	print("TRAIN: starting")
	for epoch in range(epochs):
		model.train(train_img, train_label)
	print("TRAIN: end")
	model.save()
	print("TRAIN: model saved")

TEST_LOSS = 0
TEST_ACC = 1
TRAIN_LOSS = 2
TRAIN_ACC = 3
def debug_train_command(log_record):
	config = get_config()
	batch_size = config["batch_size"]
	dataset = DatasetReader(config["dataset"])
	testing_set = dataset.load_testing()
	test_img = testing_set["images"]
	test_label = testing_set["labels"]
	training_set = dataset.load_training()
	train_img = training_set["images"]
	train_label = training_set["labels"]
	batch_img = dataset.generate_batches(training_set["images"], batch_size)
	batch_label = dataset.generate_batches(training_set["labels"], batch_size)
	model = Model(config)
	epochs = config["epochs"]
	"""
		0: testing loss
		1: testing accuracy
		2: training loss
		3: training accuracy
	"""
	results = np.zeros(epochs*4).reshape(epochs, 4)
	print("DEBUG-TRAIN: starting")
	for epoch in range(epochs):
		model.train(batch_img, batch_label)
		print(f"DEBUG-TRAIN: training completed for epoch {epoch+1}/{epochs}")
		loss, corr = model.forward_analysis(test_img, test_label)
		train_loss, train_corr = model.forward_analysis(train_img, train_label)
		results[epoch][TEST_LOSS] = loss
		results[epoch][TEST_ACC] = corr
		results[epoch][TRAIN_LOSS] = train_loss
		results[epoch][TRAIN_ACC] = train_corr
	print("DEBUG-TRAIN: end")
	model.save()
	print("DEBUG-TRAIN: model saved")
	np.savetxt(f"{C_DIR}/model_logs/{log_record}", results)
	print("DEBUG-TRAIN: debug logs saved")

def full_test_command():
	config = get_config()
	dataset = DatasetReader(config["dataset"])
	testing_set = dataset.load_testing()
	test_img = testing_set["images"]
	test_label = testing_set["labels"]
	model = Model(config)
	print("FULL-TRAIN: calculating logs")
	loss, corr = model.forward_analysis(test_img, test_label)
	print(f"| loss: {loss}")
	print(f"| accuracy: {corr*100}%")

def test_command(idx):
	config = get_config()
	dataset = DatasetReader(config["dataset"])
	testing_set = dataset.load_testing()
	test_img = dataset.generate_batches(testing_set["images"], 1)
	test_label = testing_set["labels"]
	size = len(test_img)
	if idx >= size:
		print(f"TEST: index out of range, must be less than {size}")
		return
	model = Model(config)
	res, _ = model.forward(test_img[idx])
	_, subpl = plt.subplots(1,2)
	colors = ["blue" for _ in range(len(config["outputs"]))]
	colors[test_label[idx]] = "red"
	subpl[0].imshow(test_img[idx,0], cmap="gray")
	subpl[0].axis("off")
	subpl[1].bar(config["outputs"], res[0], color=colors)
	plt.show()

def backup_command():
	FileHandler.create_backup()

def restore_command():
	FileHandler.load_backup()

def main():
	if len(sys.argv) < 2:
		print("Usage: python run.py <command>")
		print("Type \"python run.py help\" for help.")
		return
	command = sys.argv[1]
	if command == "help":
		print("python run.py <command>\n")
		print("Command list:")
		print("help : outputs this screen")
		print("init : initialises model")
		print("train : trains the model")
		print("debug-train <filename> : trains the model, logs score during train in a specified file")
		print("full-test : fully tests the model on all testing data")
		print("test <index> : tests the data on a single, specified data point")
		print("backup : creates a backup of model weights")
		print("restore : restores backed up model weights")
	elif command == "init":
		init_command()
		print("INIT: success")
	elif command == "train":
		train_command()
		print("TRAIN: success")
	elif command == "debug-train":
		if len(sys.argv) < 3:
			print("DEBUG-TRAIN: usage error")
			print("Type \"python run.py debug-train <log_name>\"")
			return
		log_record = sys.argv[2]
		debug_train_command(log_record)
		print("DEBUG-TRAIN: success")
	elif command == "full-test":
		full_test_command()
		print("FULL-TEST: success")
	elif command == "test":
		if len(sys.argv) < 3:
			print("TEST: usage error")
			print("Type \"python run.py test <index>\"")
			return
		idx = None
		try:
			idx = int(sys.argv[2])
		except:
			print("TEST: index should be an integer")
			return
		if idx < 0:
			print("TEST: index should be greater than 0")
			return
		test_command(idx)
		print("TEST: success")
	elif command == "backup":
		backup_command()
		print("BACKUP: success")
	elif command == "restore":
		restore_command()
		print("RESTORE: success")
	else:
		print("Command not recognised.")
		print("Type \"python run.py help\" for help.")





if __name__ == "__main__":
	main()