import os
import numpy as np
import matplotlib.pyplot as plt

# Display weight layers as heatmap images, shows possible training discepencies

C_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHT_FILE = lambda n: f"{C_DIR}/w{n}"
BIAS_FILE = lambda n: f"{C_DIR}/b{n}"
PARAM_FILE = lambda n: f"{C_DIR}/f{n}"




def view_stack(x, width):
	x += np.min(x)
	x /= np.max(x)
	return np.hstack([x for _ in range(width)])

WIDTH = 200
n_layer = 0
while True:
	try:
		weight_img = np.loadtxt(WEIGHT_FILE(n_layer), dtype="float64")
		bias_img = np.loadtxt(BIAS_FILE(n_layer), dtype="float64").reshape(-1,1)
		param_img = None
		subpl = None
		try:
			param_img = np.loadtxt(PARAM_FILE(n_layer), dtype="float64").reshape(-1,1)
			_, subpl = plt.subplots(1,3)
			subpl[2].imshow(view_stack(param_img, WIDTH))
			subpl[2].axis("off")
			subpl[2].set_title("function param")
		except:
			pass
		if subpl is None:
			_, subpl = plt.subplots(1,2)
		subpl[0].imshow(weight_img)
		subpl[0].axis("off")
		subpl[0].set_title("weight")
		subpl[1].imshow(view_stack(bias_img, WIDTH))
		subpl[1].axis("off")
		subpl[1].set_title("bias")
		n_layer += 1
		plt.show()
	except:
		break