## Optimization

Right now, this folder features [NonLinOptipy.py](optimization/NonLinOptipy.py). This script uses numpy, so ensure you have a proper Python environment with Numpy installed. The implemented methods in the provided NonLinOptipy.py file include Polak-Ribierre (PR) and Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithms with Armijo line search. The optimization is performed on various functions, including a quadratic function and two well-known mathematical functions: the Hump Function and the Rosenbrock function.

The quadratic function used for optimization is defined as $0.5 * x^T * Q * x - b^T * x$ where $x$ is an input vector, $Q$ is a symmetric, positive-definite matrix, and $b$ is a bias vector. The Hump Function, on the other hand, is represented by the equation $4 * x_1 ^ 2 - 2.1 * x_1 ^ 4 + x_1 ^ 6 / 3 + x_1 * x_2 - 4 * x_2 ^ 2 + 4 * x_2 ^ 4$. Lastly, the Rosenbrock function is defined as $(a - x_1) ^ 2 + b * (x_2 - x_1 ^ 2) ^ 2$, where $a$ and $b$ are constants.

The provided code demonstrates the computation of the gradient for the Rosenbrock function and the utilization of optimization methods like PR and BFGS to find the minimum of the functions. This project provides a foundation for exploring and understanding nonlinear optimization techniques and their application to specific equations.
