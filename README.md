# Neural Network Framework from Scratch
Building neural network from scratch by designing the framework only using Numpy. Inspired by the design of PyTorch library that separates layers management (forward and backward) from parameter optimization. This allows layers, losses, and optimizers to be composed cleanly.

## Project Structure
```
├── data/               
├── dev/                # Progress of building the framework (hard-written code)
├── model/              # Core neural network framework
│   ├── layers.py       # Linear, BatchNormalization, and Activation layers
│   ├── loss.py         # Loss function types
│   ├── net.py          # Sequential container and DataLoader
│   └── optimize.py     # Optimizer type
├── notebook/           # Mini experiment and demonstration
├── .gitignore
├── main.py             # Entry point for model training and evaluation
└── README.md
```

## Features
* **Modular Layers** (model.layers): Fully connected (Linear), BatchNormalization, and common activation functions.
* **Loss Functions** (model.loss): Binary and Softmax CrossEntropy (with combined logit gradient shortcuts), L1Loss, and L2Loss.
* **Optimizers** (model.optimize): Decoupled parameter updates via GradientDescent (work in progress).
* **Training Utilities** (model.net): Sequential model container and iterable DataLoader.

## Quick Start
### 1. Imports
```
from model.layers import Linear, Activation, BatchNormalization
from model.loss import SoftmaxCrossEntropy
from model.optimize import GradientDescent
from model.net import Sequential, DataLoader
```

### 2. Set Up and Initialize
```
data_load = DataLoader(X_train, y_train, batch_size=batch_size, drop_last=True, shuffle=True)

models = Sequential([
    Linear(784, 64),
    Activation('relu'),
    BatchNormalization(64),
    Linear(64, 10),
]) 
criterion = SoftmaxCrossEntropy()
optimize = GradientDescent(0.05)
```

### 3. Training Loop Execution

```
for ep in range(100):
    for batch_i, (X_batch, y_batch) in enumerate(data_load):
        output = models.forward(X_batch.T)

        loss = criterion.forward(output, y_batch.T)

        loss_grad = criterion.backward()
        grad = models.backward(loss_grad)

        parameters = models.get_params()
        optimize.step(parameters)
```

## Usage
```
pip install numpy
python main.py
```
