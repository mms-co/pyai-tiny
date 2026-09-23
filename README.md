# PyAITiny
A small, lightweight framework which can be used to build models with. This is mainly for educational purposes to show the mathematics behind models, specifically multilayer perceptrons *(as for current version)*.

Advanced frameworks such as **PyTorch** implement an **autograd** system, meaning the programmers do not need to actually know the mathematics behind backpropagation and linear regression.

---
## Features
- layers
	* standard fully connected
	* **SparseLinear**
- activation functions
	* ReLU
	* ReLU squared
	* ClippedReLU
	* tanh
	* softmax
	* sigmoid
	* SiLU
	* *(learned)* LeakyReLU
	* *(learned)* ScaledReLU
	* *(learned)* **CompositeReLU**
	* *(learned)* **DynamicReLU**
- optimisers
	* Adam optimiser
	* can be changed individually for layers and functions in backprop of `src/layers.py` and `src/funcs.py`
- normalisation
	* LayerNorm
	* RMSNorm
- initialisation
	* He distribution *(weights)*
	* constant value *(bias)*
	* easily editable inside the `src/network_init.py` file
- datasets
	* MNIST
	* Fashion-MNIST
	* more formats can be accepted by editing `src/dataset_reader.py`
- debugging/measurements
	* logging
	* training display
	* weight display

## Datasets
MNIST and Fashion-MNIST are officially implemented.
Should support other MNIST-like datasets.
Can be adapted to support other datasets.
Implemented in `src/dataset_reader.py`.

## Mathematics
### Forward Pass
The goal for a given input $X$, a probability distribution $Y$ is calculated.

#### Fully-Connected Linear
Have weights $W$ of tensor shape `(N,M)` and a bias vector $b$ with tensor shape `(1,M)`.
Take input $X$ is a vector, specifically a tensor of shape `(1,N)` (for simplicity, calculations will be for unbatched/single data).
To calculate the forward pass here, $Y = X W + b$, now vector $Y$ is a tensor of shape `(1,M)`.

#### SparseLinear
Have weights $W$ of tensor shape `(N,M)`, a bias vector $b$ with tensor shape `(1,M)`, a value $m$ and a hash function $h(x)$ which takes for input a tensor of shape `(1,N)` and returns a **constant value**. Here, it is recommended for $M$ to be large.
Take input $X$ is a vector, specifically a tensor of shape `(1,N)`.
Start by calculating $V$ as a tensor of shape `(N,m)` and $c$ as a tensor of shape `(1,m)`. To do this, gradually build with $V_i$, hash $j = h(X + i + s) \bmod m$, set $V_i = W_j$ and `c_i = b_j`.
Note that there are different ways to get a next hash, but it is important to **not** have any collisions, so hence a term $s$ was added which increments whenever $h(X + i) \bmod m$ collides with any previous selection.
Finally, calculate $Y = X V + c$, which is just a linear layer.

### Backpropagation
Having done a forward pass, $Y$ has been calculated by passing $X$ through the model, then performing a softmax operation on that final distribution. As part of the training, for any $X$ we will have a respective expected result, this result can be encoded with in a one-hot vector $Z$ as a tensor of shape `(1,M)`, same as $Y$.
As it is assumed softmax was performed, and assuming the **cross-entropy loss** function is used, $\frac{\partial L_{loss}}{\partial Y} = Y - Z$, this gradient is of shape `(1,M)`.
To be more specific, cross-entropy loss is calculated as $L_{loss} = -\sum_i Z_i \log{Y_i}$.

#### Fully-Connected Linear
Knowing weights $W$ of shape `(N,M)` and biases $b$ of shape `(1,M)` were used, we look at the forward pass $Y = X W + b$.
The model needs to learn the weight matrix $W$ and bias vector $b$, so find the derivative of each, to backpropagate the input's gradient also needs to be calculated:

$$
\frac{\partial Y}{\partial W} = X
\frac{\partial Y}{\partial b} = 1
\frac{\partial Y}{\partial X} = W
$$

These are of shapes `(1,N)`, `(1,M)` and `(N,M)` respectively.
To calculate the loss gradient for the parameters, simply apply the chain rule:

$$
\frac{\partial L_{loss}}{\partial W} = \frac{\partial Y}{\partial W} \frac{\partial L_{loss}}{\partial Y} = X^T \frac{\partial L_{loss}}{\partial Y}
\frac{\partial L_{loss}}{\partial b} = \frac{\partial Y}{\partial b} \frac{\partial L_{loss}}{\partial b} = 1 \frac{\partial L_{loss}}{\partial Y} = \frac{\partial L_{loss}}{\partial Y}
$$

Observe $X^T$, this ensures tensor shapes remain consistent, here the shapes are `(N,M)` and `(1,M)`, which match with $W$ and $b$ shapes.
Then, backpropagate to the layer input's gradient:

$$
\frac{\partial L_{loss}}{\partial X} = \frac{\partial L_{loss}}{\partial Y} \frac{\partial Y}{\partial X} = \frac{\partial L_{loss}}{\partial Y} W^T
$$

Again, observe $W^T$. Also, the order of terms the matrix multiplications which follows $Y = X W + b$, consistent both $W$ and $X$ gradient equations. The shapes remain consistent as the gradient is `(1,N)`, same as $X$.

#### SparseLinear
This is actually really similar to the Fully-Connected Layer, just cache $V$ and $c$, the gradients calculated will act on only some of $W$ and $b$.

### Parameter Update
A constant hyperparameter $\eta$ is used as the **learning rate**, this should be small.

#### Gradient Descent
This is a really simple algorithm, the learning rate is recommended to be around $\eta = 0.01$ for this.
For parameter $\theta$, gradients $G$ are calculated. Update as $\theta_t = \theta_{t-1} - \eta G_t$.


#### Adam Optimiser
There are a few flaws in gradient descent, like having no memory of previous results results, or not prioritising underperforming parameters.
A pair of hyperparameters are implemented, $\beta_1 = 0.9$ and $\beta_2 = 0.999$ (recommended values). Here, a smaller learning rate $\eta = 0.001$ is recommended.
For parameter $\theta$, gradients $G$ were calculated. First calculate respective direction and variance momentums:

$m_t = \beta_1 m_{t-1} + (1 - \beta_1) G_t$
$v_t = \beta_2 v_{t-1} + (1 - \beta2) G^2_t$

For stability, calculate relative momentum and variance capacities:

$\hat{m_t} = \frac{m_t}{1 - \beta1^t}$
$\hat{v_t} = \frac{v_t}{1 - \beta2^t}$

Now, update the gradients:

$$
\theta_t = \theta_{t-1} - \frac{\eta \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}
$$

## Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

## Installation
```bash
git clone https://github.com/mms-co/pyai-tiny
cd pyai-tiny
pip install -r requirements.txt
cd src
```

## Usage
A pre-made example model is already made inside `model.py`, to quick-start just initialise, train and test the model:
- Start training
```bash
python run.py train
```
- Start debug training
```bash
python run.py debug-train [LOG_NAME]
```
- Initialise model
```bash
python run.py init
```
- Complete full dataset test
```bash
python run.py full-test
```
- Specified test as index from test dataset
```bash
python run.py test [INDEX]
```
- View model weights
```bash
python network_data/param_display.py
```
- View a debug training log for all stats
```bash
python model_logs/log_display.py single [LOG_NAME]
```
- Compare debug trainings for a stat
```bash
python model_logs/log_display.py compare [STAT] [LOG_NAME1] [LOG_NAME2] ...
```
- Weights backup
```bash
python run.py backup
```
- Load backup
```bash
python run.py restore
```

To edit the model, change the configs in `config.json`, change the model architecture in `model.py`, change the initialisation methods in `components/network_init.py` while other configs can be set in `run.py`, change individual optimisers or add new implementatoions in `components/funcs.py` and `components/layers.py`.

### Model
Inside `model.py`, there is a `Model` class where the architecture is implemented.
```py
class Model:
	def __init__(self, settings):
		# Initialise all model layers here
	def forward(self, input_layer):
		# Forward pass, must return calculation caches for backpropagation
	def backprop(self, batch_input, expected_output):
		# Run forward pass
		# Show how derivative propagates, this is where caching was needed
	# Other methods don't need to be modified
```

## License
Distributed under the MIT License. See `LICENSE` for more information.