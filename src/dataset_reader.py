import numpy as np
import struct
from array import array
from matplotlib import pyplot
import os

C_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_SCOPE = C_DIR + "/../"
TRAIN_IMG = lambda db: DATA_SCOPE + f"datasets/{db}/train/img/"
TRAIN_LABEL = lambda db: DATA_SCOPE + f"datasets/{db}/train/label/"
TEST_IMG = lambda db: DATA_SCOPE + f"datasets/{db}/test/img/"
TEST_LABEL = lambda db: DATA_SCOPE + f"datasets/{db}/test/label/"



class DatasetReader:
	def __init__(self, dataset: str):
		self.__dataset = dataset
	def __read_label(self, path: str, magic_number: int = None):
		labels = []
		with open(path, "rb") as file:
			magic, size = struct.unpack(">II", file.read(8))
			if magic_number != None and magic != magic_number:
				raise ValueError(f"Magic number mismatch, {magic} is not {magic_number}.")
			label_data = array("B", file.read())
			labels = np.array(label_data, dtype=np.ubyte)
			file.close()
		return labels
	# MNIST-like image loader
	def __read_monoimage(self, path: str, metadata: tuple[int, int],
			magic_number: int = None):
		images = []
		with open(path, "rb") as file:
			magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
			if magic_number != None and magic != magic_number:
				raise ValueError(f"Magic number mismatch, {magic} is not {magic_number}.")
			image_data = array("B", file.read())
			imgs: list = []
			for i in range(size):
				img = np.array(image_data[i*rows*cols:(i+1)*rows*cols],
					dtype=np.ubyte)
				# Normalise pixels (0-255)
				img = img.reshape(metadata).astype(np.float64) / 255.
				imgs.append(img)
			images = np.array(imgs, dtype=np.float64)
			file.close()
		return images

	# Dataset loaders (train/test)
	def __load_mnist_train(self) -> tuple:
		MAGIC_IMG: int = 2051
		IMAGE_FILE: str = "train-images.idx3-ubyte"
		IMAGE_PATH: str = TRAIN_IMG("mnist_digits") + IMAGE_FILE
		IMAGE_SIZE: tuple[int] = (28, 28)
		images = self.__read_monoimage(IMAGE_PATH, IMAGE_SIZE, MAGIC_IMG)
		MAGIC_LABEL: int = 2049
		LABEL_FILE: str = "train-labels.idx1-ubyte"
		LABEL_PATH: str = TRAIN_LABEL("mnist_digits") + LABEL_FILE
		labels = self.__read_label(LABEL_PATH, MAGIC_LABEL)
		return images, labels
	def __load_mnist_test(self) -> tuple:
		MAGIC_IMG: int = 2051
		IMAGE_FILE: str = "t10k-images.idx3-ubyte"
		IMAGE_PATH: str = TEST_IMG("mnist_digits") + IMAGE_FILE
		IMAGE_SIZE: tuple[int] = (28, 28)
		images = self.__read_monoimage(IMAGE_PATH, IMAGE_SIZE, MAGIC_IMG)
		MAGIC_LABEL: int = 2049
		LABEL_FILE: str = "t10k-labels.idx1-ubyte"
		LABEL_PATH: str = TEST_LABEL("mnist_digits") + LABEL_FILE
		labels = self.__read_label(LABEL_PATH, MAGIC_LABEL)
		return images, labels


	def __load_fashion_mnist_train(self) -> tuple:
		MAGIC_IMG: int = 2051
		IMAGE_FILE: str = "train-images-idx3-ubyte"
		IMAGE_PATH: str = TRAIN_IMG("fashion_mnist") + IMAGE_FILE
		IMAGE_SIZE: tuple[int] = (28, 28)
		images = self.__read_monoimage(IMAGE_PATH, IMAGE_SIZE, MAGIC_IMG)
		MAGIC_LABEL: int = 2049
		LABEL_FILE: str = "train-labels-idx1-ubyte"
		LABEL_PATH: str = TRAIN_LABEL("fashion_mnist") + LABEL_FILE
		labels = self.__read_label(LABEL_PATH, MAGIC_LABEL)
		return images, labels
	def __load_fashion_mnist_test(self) -> tuple:
		MAGIC_IMG: int = 2051
		IMAGE_FILE: str = "t10k-images-idx3-ubyte"
		IMAGE_PATH: str = TEST_IMG("fashion_mnist") + IMAGE_FILE
		IMAGE_SIZE: tuple[int] = (28, 28)
		images = self.__read_monoimage(IMAGE_PATH, IMAGE_SIZE, MAGIC_IMG)
		MAGIC_LABEL: int = 2049
		LABEL_FILE: str = "t10k-labels-idx1-ubyte"
		LABEL_PATH: str = TEST_LABEL("fashion_mnist") + LABEL_FILE
		labels = self.__read_label(LABEL_PATH, MAGIC_LABEL)
		return images, labels


	def generate_batches(self, training_data, batch_size):
		shape = training_data.shape
		length = shape[0]
		# Array is divided into chunks, batch_size must divide length as a consequence
		if length / batch_size != int(length / batch_size):
			raise ValueError(f"Cannot form batches, {length} not divisible by {batch_size}")
		return training_data.reshape((length // batch_size, batch_size)+shape[1:])

	def load_training(self) -> dict:
		result = {}
		if self.__dataset == "mnist":
			imgs, labels = self.__load_mnist_train()
			result["images"] = imgs
			result["labels"] = labels
		elif self.__dataset == "fashion_mnist":
			imgs, labels = self.__load_fashion_mnist_train()
			result["images"] = imgs
			result["labels"] = labels
		return result
	def load_testing(self) -> dict:
		result = {}
		if self.__dataset == "mnist":
			imgs, labels = self.__load_mnist_test()
			result["images"] = imgs
			result["labels"] = labels
		elif self.__dataset == "fashion_mnist":
			imgs, labels = self.__load_fashion_mnist_test()
			result["images"] = imgs
			result["labels"] = labels
		return result

# Quick test, check if it loads
if __name__ == "__main__":
	reader = DatasetReader("fashion_mnist")
	reader.load_training()
	t = reader.load_testing()
	print(t['labels'][4])
	for img in t["images"]:
		pyplot.imshow(img)
		pyplot.show()