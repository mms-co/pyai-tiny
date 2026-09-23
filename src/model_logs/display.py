import numpy as np
import matplotlib.pyplot as plt
import sys
import os

C_DIR = os.path.dirname(os.path.abspath(__file__))

path = lambda n: f"{C_DIR}/{n}"

# Display analysis data as a graph

TEST_LOSS = 0
TEST_ACC = 1
TRAIN_LOSS = 2
TRAIN_ACC = 3

ACC = 0
LOSS = 1

# Min/Max point plots
def plot_min(plot, x, y):
	min_idx = np.argmin(y)
	min_x = x[min_idx]
	min_y = y[min_idx]
	plot.annotate(
		f"min ({min_x}, {min_y})"
		, xy=(min_x,min_y)
		, xytext=(15,-5)
		, textcoords="offset points"
	)
	plot.scatter(min_x, min_y, marker='o', zorder=5)

def plot_max(plot, x, y):
	max_idx = np.argmax(y)
	max_x = x[max_idx]
	max_y = y[max_idx]
	plot.annotate(
		f"max ({max_x}, {max_y})"
		, xy=(max_x,max_y)
		, xytext=(15,-5)
		, textcoords="offset points"
	)
	plot.scatter(max_x, max_y, marker='o', zorder=5)


# # One-graph multi-line display
# def display(dataset):
# 	path = lambda t: dataset + '/data-' + t
# 	for d in DATABASE:
# 		data = np.loadtxt(path(d), dtype='float64')[:,DataPoints.test_loss]
# 		x = np.arange(len(data))
# 		plt.plot(x, data, label=d)
# 		plot_min(x, data)
# 	plt.legend()
# 	plt.show()


# display('mnist')

def main():
	if len(sys.argv) < 2:
		print("Usage: python display.py <command>")
		print("Type \"python display.py help\" for help.")
		return
	command = sys.argv[1]
	if command == "help":
		print("python display.py <command>\n")
		print("Command list:")
		print("help : outputs this screen")
		print("single <filename> : shows evaluation log for one log file")
		print("compare <stat> <filenames> : shows multiple evaluation logs for one statistic (test-loss/test-acc/train-loss/train-acc)")
	elif command == "single":
		if len(sys.argv) < 3:
			print("SINGLE: must provide a log file")
			return
		log_record = sys.argv[2]
		try:
			data = np.loadtxt(path(log_record))
			accuracy = data[:,[TEST_ACC, TRAIN_ACC]]
			loss = data[:,[TEST_LOSS, TRAIN_LOSS]]
			epochs = np.arange(data.shape[0])
			_, subpl = plt.subplots(1,2)
			# Accuracy
			subpl[ACC].plot(epochs, accuracy[:,0], label="test")
			plot_max(subpl[ACC], epochs, accuracy[:,0])
			subpl[ACC].plot(epochs, accuracy[:,1], label="train")
			plot_max(subpl[ACC], epochs, accuracy[:,1])
			subpl[ACC].legend()
			# Loss
			subpl[LOSS].plot(epochs, loss[:,0], label="test")
			plot_min(subpl[LOSS], epochs, loss[:,0])
			subpl[LOSS].plot(epochs, loss[:,1], label="train")
			plot_min(subpl[LOSS], epochs, loss[:,1])
			subpl[LOSS].legend()
			plt.show()
		except:
			print("SINGLE: file not found, or formatted incorrectly")
	elif command == "compare":
		if len(sys.argv) < 3:
			print("COMPARE: must provide a statistic to compare")
			print("test-loss / test-acc / train-loss / train-acc")
			return
		OPTIONS = {
			"test-loss": TEST_LOSS,
			"test-acc": TEST_ACC,
			"train-loss": TRAIN_LOSS,
			"train-acc": TRAIN_ACC
		}
		stat = None
		try:
			stat = OPTIONS[sys.argv[2]]
		except:
			print("COMPARE: invalid statistic")
			print("test-loss / test-acc / train-loss / train-acc")
			return
		point_plotter = plot_max if stat < 2 else plot_min
		if len(sys.argv) < 4:
			print("COMPARE: must provide which log files to compare")
			return
		files = sys.argv[3:]
		try:
			for file in files:
				data = np.loadtxt(path(file))[:,stat]
				epochs = np.arange(data.shape[0])
				plt.plot(epochs, data, label=file)
				point_plotter(plt, epochs, data)
			plt.legend()
			plt.show()
		except:
			print("COMPARE: a file is missing, or is formatted incorrectely")
	else:
		print("Command not recognised.")
		print("Type \"python display.py help\" for help.")

if __name__ == "__main__":
	main()