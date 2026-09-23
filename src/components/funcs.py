import numpy as np


class ReLU:
	def forward(x):
		return np.maximum(0, x)
	def back(x, deriv):
		return (x > 0) * deriv

class ClippedReLU:
	def __init__(self, ceiling=1):
		self.ceiling = ceiling
	def forward(self, x):
		return np.minimum(self.ceiling, np.maximum(0, x))
	def back(self, x, deriv):
		return ((x < 0) & (x > self.ceiling)) * deriv

class Softmax:
	def forward(x):
		# Needed numeric stability
		e = np.exp(x - np.max(x, axis=1, keepdims=True))
		return e / np.sum(e, axis=1, keepdims=True)
	def back(x, deriv):
		y = Softmax.forward(x)
		dx = np.zeros_like(x)
		batch_size = x.shape[0]
		for i in range(batch_szie):
			y_batch = y[i].shape(-1, 1)
			j = np.diagflat(y_batch) - np.dot(y_batch, y_batch.T)
			dx[i] = np.dot(j, deriv[i])
		return dx

class Tanh:
	def forward(x):
		ex = np.exp(2 * x)
		return (ex - 1) / (ex + 1)
	def back(x, deriv):
		px = np.exp(x)
		nx = np.exp(-x)
		return deriv * 4 / (px + nx)**2

class Sigmoid:
	def forward(x):
		return 1 / (1 + np.exp(-x))
	def back(x, deriv):
		ex = np.exp(-x)
		return deriv * ex / (1 + ex)**2

class SiLU:
	def forward(x):
		return x / (1 + np.exp(-x))
	def back(x, deriv):
		ex = np.exp(-x)
		return ( x * ex/(1 + ex)**2 + 1/(1 + ex) ) * deriv

# Warning: ReLU^2 often overflows
class ReLUsq:
	def forward(x):
		return np.maximum(0, x)**2
	def back(x, deriv):
		dy = np.copy(x)
		dy[dy < 0] = 0
		return 2 * dy * deriv

# These need initialisation - extra parameters used
# This also means extra backprop needed to update the paramters
# Parameters transfer by reference
class LeakyReLU:
	def __init__(self, param):
		self.param = param
	def forward(self, x):
		batch_size = x.shape[0]
		y = np.copy(x)
		batch_a = np.tile(self.param, (batch_size,1))
		mapping = y < 0
		y[mapping] *= batch_a[mapping]
		return y
	def back(self, x, deriv,
		lr=0.01,
		decay=1,
		batch_size=50
	):
		# Calculate grads
		dy = np.copy(x)
		batch_size = x.shape[0]
		batch_da = np.tile(self.param, (batch_size, 1))
		mapping = dy < 0
		dy[mapping] = batch_da[mapping]
		dy[~mapping] = 1
		batch_da[mapping] = x[mapping]
		batch_da[~mapping] = 0
		da = np.mean(batch_da * deriv, axis=0)

		# Update params: a=SGD
		self.param -= decay*lr * da
		return dy * deriv

class ScaledReLU:
	def __init__(self, param):
		self.param = param
	def forward(self, x):
		batch_size = x.shape[0]
		y = np.copy(x)
		batch_b = np.tile(self.param, (batch_size,1))
		mapping = y > 0
		y[mapping] *= batch_b[mapping]
		return y
	def back(self, x, deriv,
		lr=0.01,
		decay=1,
		batch_size=50
	):
		# Calculate grads
		dy = np.copy(x)
		batch_size = x.shape[0]
		batch_db = np.tile(self.param, (batch_size, 1))
		mapping = dy > 0
		dy[mapping] = batch_db[mapping]
		dy[~mapping] = 1
		batch_db[mapping] = x[mapping]
		batch_db[~mapping] = 0
		db = np.mean(batch_db * deriv, axis=0)

		# Update params: b=Adam
		self.param -= decay*lr * db
		return dy * deriv

class CompositeReLU:
	def __init__(self, param_a, param_b):
		self.param_a = param_a
		self.param_b = param_b
	def forward(self, x):
		batch_size = x.shape[0]
		y = np.copy(x)
		batch_a = np.tile(self.param_a, (batch_size,1))
		batch_b = np.tile(self.param_b, (batch_size,1))
		mapping = y < 0
		y[mapping] *= batch_a[mapping]
		y[~mapping] *= batch_b[~mapping]
		return y
	def back(self, x, deriv,
		lr=0.01,
		decay=1,
		batch_size=50
	):
		# Calculate grads
		dy = np.copy(x)
		batch_size = x.shape[0]
		batch_da = np.tile(self.param_a, (batch_size, 1))
		batch_db = np.tile(self.param_b, (batch_size, 1))
		mapping = dy > 0
		dy[mapping] = batch_db[mapping]
		dy[~mapping] = batch_da[~mapping]
		batch_db[mapping] = x[mapping]
		batch_da[~mapping] = x[~mapping]
		db = np.mean(batch_db * deriv, axis=0)
		da = np.mean(batch_da * deriv, axis=0)

		# Update params: a=SGD, b=SGD
		self.param_a -= decay*lr * da
		self.param_b -= decay*lr * db
		return dy * deriv

class DynamicReLU:
	def __init__(self, param_a, param_b, param_beta):
		self.param_a = param_a
		self.param_b = param_b
		self.param_beta = param_beta
	def forward(self, x):
		batch_size = x.shape[0]
		y = np.copy(x)
		batch_a = np.tile(self.param_a, (batch_size,1))
		batch_b = np.tile(self.param_b, (batch_size,1))
		batch_beta = np.tile(self.param_beta, (batch_size,1))
		mapping = y <= self.param_beta
		y[mapping] *= batch_a[mapping]
		y[~mapping] = batch_a[~mapping] * batch_beta[~mapping] + batch_b[~mapping] * (y[~mapping] - batch_beta[~mapping])
		return y
	def back(self, x, deriv,
		lr=0.01,
		decay=1,
		batch_size=50
	):
		# Calculate grads
		batch_size = x.shape[0]
		dy = np.copy(x)
		batch_da = np.tile(self.param_a, (batch_size,1))
		temp_a = np.copy(batch_da)
		batch_db = np.tile(self.param_b, (batch_size,1))
		temp_b = np.copy(batch_db)
		batch_dbeta = np.tile(self.param_beta, (batch_size,1))
		mapping = dy <= self.param_beta
		dy[mapping] = batch_da[mapping]
		dy[~mapping] = batch_db[~mapping]
		batch_da[mapping] = x[mapping]
		batch_da[~mapping] = batch_dbeta[~mapping]
		batch_db[mapping] = 0
		batch_db[~mapping] = x[~mapping] - batch_dbeta[~mapping]
		batch_dbeta[mapping] = 0
		batch_dbeta[~mapping] = temp_a[~mapping] - temp_b[~mapping]
		da = np.mean(batch_da * deriv, axis=0)
		db = np.mean(batch_db * deriv, axis=0)
		dbeta = np.mean(batch_dbeta * deriv, axis=0)

		# Update params: a=SGD, b=SGD, beta=SGD
		self.param_a -= decay*lr * da
		self.param_b -= decay*lr * db
		self.param_beta -= decay*lr * dbeta
		return dy * deriv


# Util functions
def cross_entropy_loss(predicted, result,
		eps=1e-8):
	out = np.clip(predicted, eps, 1-eps)
	losses = -np.sum(result * np.log(out), axis=1)
	return np.mean(losses)

# Cross entropy derivative, never needed more than this
def cost_deriv(output, target):
	return output - target