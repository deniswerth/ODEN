[![Python](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.2-orange.svg)](https://tensorflow.org)
# Neural Network ODEsolver
| &nbsp;        | &nbsp;           |
| ------------- |:-------------:|
|**NN ODEsolver:**  | Code that numerically solves ODEs using Neural Networks in an unsupervised manner|
| **Authors:**  |[Liam L.H. Lau](https://github.com/LiamLau1), [Denis Werth](https://github.com/deniswerth)|
| **Version:**  | 1.0|
| **Homepage:**  | [https://github.com/deniswerth/NeuralNetwork_ODEsolver](https://github.com/deniswerth/NeuralNetwork_ODEsolver)|
## Motivations
Feedforward neural networks are able to approximate any continuous function at any level of accuracy. This is a loose statement of the Universal Approximation Theorem for neural networks. Although there is open source code for neural network solvers for ODEs and PDEs, the difference is that this network trains to fit the differential equation and boundary/ initial conditions at the same time. We also suggest that the loss is approximately the mean absolute error, without the need of having the true solution. 

The paper can be found here:

All examples from the paper can be reproduced with the code provided, a walk through is provided in the examples section below.

## Usage
### Main code in `__Main__.py`:
* To replicate the plots in the paper and in **Examples** below, comment out all the `if __name__ == "__main__":` sections except the particular example you want to keep.

### Neural Network Solver class in `ODEsolver.py`:
Imported in `__Main__.py` by:

```python
from ODEsolver import ODEsolver
```

### Dictionary class in `Dictionary.py`:
Imported in `__Main__.py` by:

```python
from Dictionary import Dictionary
D = Dictionary()
Dict = D.Dict
```

* Dictionary class that contains the initializers, activation functions and optimizers that 

### Differential equation class in `DiffEq.py`:
Imported in `__Main__.py` by:

```python
from DiffEq import DiffEq
```

* Differential equation class that contains definition of differential equation problem. Three examples are given: a first order ODE; \\(n^{th}\\) stationary Schrodinger equation with a harmonic potential and a burst equation.
* A custom differential equation can be defined as the following in `DiffEq.py`:
```python
if self.diffeq == "name":
    # x should be written as self.x; y should be written as self.y; dy/dx should be written as self.dydx and d2y/dx2 should be written as self.d2ydx2
    self.eq = The custom differential equation 

```
* Currently, the code only supports up to second order. However higher orders can be implemented by editing the `y_gradients` function for the Neural Network Solver class in `ODEsolver.py`. For example, third order can be accessed by using the following code snippet:
```python
def y_gradients(self, x):
    """
    Computes the gradient of y.
    """
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape3:
                tape3.watch(x)
                y = self.NN_output(x)
            dy_dx = tape3.gradient(y, x)
        d2y_dx2 = tape2.gradient(dy_dx,x)
    d3y_dx3 = tape1.gradient(d2y_dx2,x)
    return y, dy_dx, d2y_dx2, d3y_dx3
```
Make sure to also edit the `differential_cost` function for the Neural Network Solver class in `ODEsolver.py` to include the extra orders:

```python
y, dydx, d2ydx2, d3ydx3 = self.y_gradients(x)
```

```python
de = DiffEq(self.diffeq, x, y, dydx, d2ydx2, d3ydx3)
```
The third order derivative will now be available in `DiffEq.py` using `self.d3ydx3`

## Examples
