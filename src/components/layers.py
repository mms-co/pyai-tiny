import numpy as np

class Linear:
	def __init__(self, weight, bias):
		self.weight = weight
		self.bias = bias
		self.cache = None
	def forward(self, x):
		res = x @ self.weight + self.bias
		self.cache = np.copy(x)
		return res
	def back(self, deriv,
		lr=0.01,
		decay=1,
		batch_size=50
	):
		# Calculate grads
		y = self.cache
		dweight = y.T @ deriv
		dbias = np.sum(deriv, axis=0)
		new_deriv = deriv @ self.weight.T

		# Update params: weight=SGD, bias=SGD
		self.weight -= decay*lr * dweight / batch_size
		self.bias -= decay*lr * dbias / batch_size
		return new_deriv

class LayerNorm:
	def __init__(self, shift, scale):
		self.shift = shift
		self.scale = scale
		self.cache = ()

		self.momentum_h = np.zeros_like(self.shift)
		self.variance_h = np.zeros_like(self.shift)
		self.momentum_c = np.zeros_like(self.scale)
		self.variance_c = np.zeros_like(self.scale)
	def forward(self, x, eps=1e-8):
		m = np.mean(x, axis=1, keepdims=True)
		v = np.var(x, axis=1, keepdims=True)
		norm = (x - m) / np.sqrt(v + eps)
		self.cache = (x.shape[1], v, norm)
		return norm * self.scale + self.shift
	def back(self, deriv,
		update_iter,
		lr=0.0001,
		decay=1,
		beta1=0.95,
		beta2=0.999,
		eps=1e-8
	):
		# Calcualte grads
		n, v, norm = self.cache
		std_inv = 1. / np.sqrt(v + eps)
		dscale = np.sum(deriv * norm, axis=0, keepdims=True)
		dshift = np.sum(deriv, axis=0, keepdims=True)
		dnorm = deriv * self.scale
		grad = (1. / n) * std_inv * (
			n * dnorm
			- np.sum(dnorm, axis=1, keepdims=True)
			- norm * np.sum(dnorm * norm, axis=1, keepdims=True)
		)

		# Update params: scale=Adam, shift=Adam
		momentum_capacity = 1 - beta1**update_iter
		variance_capacity = 1 - beta2**update_iter
		self.momentum_h = beta1 * self.momentum_h + (1 - beta1) * dshift
		momentum_hat = self.momentum_h / momentum_capacity
		self.variance_h = beta2 * self.variance_h + (1 - beta2) * dshift**2
		variance_hat = self.momentum_h / variance_capacity
		self.shift -= decay*lr * momentum_hat / (np.sqrt(variance_hat) + eps)

		self.momentum_c = beta1 * self.momentum_c + (1 - beta1) * dscale
		momentum_hat = self.momentum_c / momentum_capacity
		self.variance_c = beta2 * self.variance_c + (1 - beta2) * dscale**2
		variance_hat = self.variance_c / variance_capacity
		self.scale -= decay*lr * momentum_hat / (np.sqrt(variance_hat) + eps)
		return grad


class RMSNorm:
	def __init__(self, scale):
		self.scale = scale
		self.cache = ()

		self.momentum = np.zeros_like(self.scale)
		self.variance = np.zeros_like(self.scale)
	def forward(x, scale, eps=1e-8):
		rms = np.sqrt( np.mean(x**2, axis=1, keepdims=True) + eps )
		norm = x / rms
		self.cache = (x.shape[1], norm, rms)
		return norm * self.scale
	def back(self, deriv,
		update_iter,
		lr=0.0001,
		decay=1,
		beta1=0.95,
		beta2=0.999,
		eps=1e-8
	):
		# Calculate grads
		n, norm, rms = self.cache
		dscale = np.sum(deriv * norm, axis=0, keepdims=True)
		dnorm = deriv * self.scale
		grad = 1. / rms * (
			dnorm - norm
			* np.mean(dnorm * norm, axis=1, keepdims=True)
		)

		# Update params: scale=Adam
		self.momentum = beta1 * self.momentum + (1 - beta1) * dscale
		momentum_hat = self.momentum / (1 - beta1**update_iter)
		self.varaince = beta2 * self.variance + (1 - beta2) * dscale**2
		variance_hat = self.variance / (1 - beta2**update_iter)
		self.scale -= decay*lr * momentum_hat / (np.sqrt(variance_hat) + eps)
		return grad

class Dropout:
	def __init__(self, dropout=0.2):
		self.keep = 1 - dropout
		self.cache = None
	def forward(self, x, train=True):
		if not train:
			return x
		drop = (np.random.rand(*x.shape) < self.keep) / self.keep
		self.cache = drop
		return x * drop
	def back(self, deriv):
		drop = self.cache
		return deriv * drop

# This takes 2 layers, as sparsity data affects the next layer
class SparseLinear:
	def __init__(self, sparse_neurons, activation, neurons,
			hashf, count):
		self.sparse_weight, self.sparse_bias = sparse_neurons
		self.activation = activation
		self.weight, self.bias = neurons
		# hashf : np.ndarray -> int
		self.hashf = hashf
		self.count = count
		self.cache = ()
	def forward(self, x):
		dim = self.sparse_weight.shape[1]
		# Hash-based bucketing system
		bucket = set()
		shift = 1
		while len(bucket) != self.count:
			elem = self.hashf(x * shift) % dim
			if elem not in bucket:
				bucket.add(elem)
			shift += 1
		bucket = list(bucket)
		bucket = sorted(bucket)
		# Select bucket for this layer
		weight_bucket = self.sparse_weight[:,bucket]
		bias_bucket = self.sparse_bias[bucket]
		sparse_res = x @ weight_bucket + bias_bucket
		act = self.activation.forward(sparse_res)
		# Select bucket for next layer
		next_weight = self.weight[bucket]
		res = sparse_res @ next_weight + self.bias
		self.cache = (np.copy(x), sparse_res, bucket)
		return res
	def back(self, deriv,
		lr=0.01,
		decay=1,
		batch_size=50
	):
		# Calculate grads
		y, z, bucket = self.cache
		# Get buckets, easy access as data is kept in cache
		weight_bucket = self.sparse_weight[:,bucket]
		next_weight = self.weight[bucket]
		bias_bucket = self.sparse_bias[bucket]
		dweight = z.T @ deriv
		dbias = np.sum(deriv, axis=0)
		new_deriv = deriv @ next_weight.T

		new_deriv = self.activation.back(z, new_deriv)

		dsparse_weight = y.T @ new_deriv
		dsparse_bias = np.sum(new_deriv, axis=0)
		new_deriv = new_deriv @ weight_bucket.T

		# Update params: weight=SGD, bias=SGD
		self.sparse_weight[:,bucket] -= decay*lr * dsparse_weight / batch_size
		self.sparse_bias[bucket] -= decay*lr * dsparse_bias / batch_size
		self.weight[bucket] -= decay*lr * dweight / batch_size
		self.bias -= decay*lr * dbias / batch_size
		return new_deriv



# Used for muon optimiser
def muon(momentum, steps=5, eps=1e-8):
	a, b, c = 3.4445, -4.7750, 2.0315
	norm = momentum / (np.linalg.norm(momentum) + eps)
	transposed = momentum.shape[0] > momentum.shape[1]
	if transposed:
		norm = norm.T
	for _ in range(steps):
		A = norm @ norm.T
		B = b * A + c * A @ A
		norm = a * norm + B @ norm
	if transposed:
		norm = norm.T
	return norm