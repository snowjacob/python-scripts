import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Implementing Test Functions')
parser.add_argument('--Q', type=str, default='[[15, 14], [16, 18]]', help='Q matrix (Positive Semidefinite')
parser.add_argument('--b', type=str, default='[1, 2]', help='b vector')
parser.add_argument('--x', type=str, default='[1, -1]', help='x vector')
args = parser.parse_args()

Q = np.array(eval(args.Q))
b = np.array(eval(args.b))
x = np.array(eval(args.x))

# Takes in a vector x, symmetric matrix Q, and bias vector b. Assumes dimensions are compatible
def myQuad(x, Q, b):
    # r: scalar value of the quadratic function evaluated at x
    r = 0.5 * x.T @ Q @ x - b.T @ x
    # g: gradient of the quadratic function evaluated at x (assuming symmetry)
    g = Q @ x - b
    return r, g

# Since good tests come up when b >= 100a > 0, we will choose b = 250, a = 2. Assumes x is 2-dimensional
def myRosenbrock(x):
    a = 2
    b = 250
    # Computes Rosenbrock function at x
    r = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
    # Computes gradient of Rosenbrock function at x
    g = np.array([-2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2),
                  2 * b * (x[1] - x[0] ** 2)])
    return r, g

# Returns a valid step size maxa for minimizing f(x) using a descent algorithm
def myArmijo(x, f, p, c, maxa, r):
    v, g = f(x)
    while f(x + maxa * p)[0] > v + c * maxa * p @ g:
        maxa = r * maxa
    return maxa

# Takes in a vector x, function f, and tolerance and returns the minimum
def myBFGS(x, f, tol):
    H = np.eye(len(x))
    runs = 0
    x_array = np.empty([10, 2])
    x_array[0] = x
    while runs < 10:
        g = f(x)[1]
        if np.sqrt(g @ g.T) < tol or runs == 4:
            return x, f(x)[0], np.sqrt(g @ g.T), x_array
        p = -1 * H @ g.T
        a = myArmijo(x, f, p, 0.5, 0.01, 0.5)
        newx = x + a * p
        delta_x = newx - x
        delta_g = f(newx)[1] - g
        # IF THE DENOMINATOR BECOMES ZERO, RETURN (This usually happens when delta_g is zero)
        if delta_g @ delta_x.T == 0:
            return x, f(x)[0], np.sqrt(g @ g.T), x_array
        H = H + (1 + (delta_g @ H @ delta_g.T) / (delta_g @ delta_x.T)) * (
                    (delta_x @ delta_x.T) / (delta_x @ delta_g.T)) - (
                    (H @ delta_g.T @ delta_x) + (H @ delta_g.T @ delta_x).T) / (delta_g @ delta_x.T)
        x_array[runs + 1] = newx
        x = newx
        runs += 1

# Takes in a vector x, function f, and tolerance and returns the minimum. Assumes f is quadratic
# Polak-Ribierre
def myPR(x, f, tol):
    x_array = np.empty([10, 2])
    x_array[0] = x
    runs = 0
    p = -f(x)[1]
    while runs < 10:
        g = f(x)[1]
        if np.sqrt(g @ g.T) < tol or runs == 4:
            return x, f(x)[0], np.sqrt(g @ g.T), x_array
        a = myArmijo(x, f, p, 0.5, 0.01, 0.5)
        newx = x + a * p
        beta = (f(newx)[1] @ (f(newx)[1] - g).T) / (g @ g.T)
        p = -1 * f(newx)[1] + beta * p
        x_array[runs + 1] = newx
        x = newx
        runs += 1

# Hump Function
def myHump(x):
    r = 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + x[0] ** 6 / 3 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4
    g = np.array([8 * x[0] - 8.4 * x[0] ** 3 + 2 * x[0] ** 5 + x[1], x[0] - 8 * x[1] ** 3 + 16 * x[1] ** 3])
    return r, g

f = lambda x: myQuad(x, Q, b)
q = lambda x: myRosenbrock(x)
h = lambda x: myHump(x)

tol = 10 ** -2

# CODE USED IN ANALYSIS

[newx, nfunc, gradnorm, x_array] = myPR(x, q, tol)
# [newx, nfunc, gradnorm, x_array] = myBFGS(x, q, tol)
# [newx, nfunc, gradnorm, x_array] = myBFGS(x, f, tol)
# [newx, nfunc, gradnorm, x_array] = myPR(x, f, tol)
# [newx, nfunc, gradnorm, x_array] = myPR(x, h, tol)

for i in range(len(x_array)):
    print(x_array[i])
    print(h(x_array[i])[0])
