import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description='Implementing Test Functions')
parser.add_argument('--Q', type=str, default=np.ones((2, 2)), help='Q matrix (Positive Semidefinite')
parser.add_argument('--b', type=str, default=np.ones(2), help='b vector')
parser.add_argument('--x', type=str, default=np.ones(2), help='x vector')
args = parser.parse_args()

Q = args.Q
b = args.b
x = args.x

Q = np.array([[10, 30], [43, 20]])
x = np.array([12, 12])
b = np.array([1, 2])

#Takes in a vector x, symmetric matrix Q, and bias vector b. Assumes dimensions are compatible
def myQuad(x, Q, b):
    # r : scalar value of the quadratic function evaluated at x
    r = 0.5 * x.T @ Q @ x - b.T @ x
    # g : gradient of the quadratic function evaluated at x (assuming symmetry)
    g = Q @ x - b
    return [r, g]

#Since good tests come up when b >= 100a > 0, we will choose b = 250, a = 2. Assumes x is 2-dimensional
def myRosenbrock(x):
    x = np.array(x)
    if(x.size != 2): return "Error: x must be 2-dimensional"
    a = 2
    b = 250
    #Computes Rosenbrock function at x
    r = math.pow((a - x[0]), 2) + b * math.pow((x[1] - math.pow(x[0], 2)), 2)
    #Computes gradient of Rosenbrock function at x
    g = np.array([-2 * (a - x[0]) - 4 * b * x[0] * (x[1] - math.pow(x[0], 2)), 2 * b * (x[1] - math.pow(x[0], 2))])
    return [r, g]


#REVISION: Should've been p @ g.T instead of p @ g
#Returns a valid step size maxa for minimizing f(x) using a descent algorithm
def myArmijo(x, f, p, c, maxa, r):
    [v, g] = f(x)
    while(f(x + maxa * p)[0] > v + c * maxa * p @ g.T):
        maxa = r * maxa
    return maxa

#Takes in a vector x, function f, and tolerance and returns the minimum
def myBFGS(x, f, tol):
    H = np.eye(len(x))
    runs = 0
    while (runs < 5):
        g = f(x)[1]
        if(np.sqrt(g @ g.T) < tol or runs == 4):
            return [x, f(x)[0], np.sqrt(g @ g.T)]
        p = -1 * H @ g.T
        a = myArmijo(x, f, p, 0.5, 0.01, 0.5)
        newx = x + a * p
        delta_x = newx - x
        delta_g = f(newx)[1] - g
        #IF THE DENOMINATOR BECOMES ZERO, RETURN (This usually happens when delta_g is zero)
        if(delta_g @ delta_x.T == 0):
            return [x, f(x)[0], np.sqrt(g @ g.T)]
        H = H + (1 + (delta_g @ H @ delta_g.T) / (delta_g @ delta_x.T)) * ((delta_x @ delta_x.T)/(delta_x @ delta_g.T)) - (((H @ delta_g.T @ delta_x) + (H @ delta_g.T @ delta_x).T) / (delta_g @ delta_x.T))
        x = newx
        runs += 1

#Takes in a vector x, function f, and tolerance and returns the minimum. Assumes f is quadratic
def myPR(x, f, tol):
    runs = 0
    p = -f(x)[1]
    while(runs < 5):
        print(runs)
        g = f(x)[1]
        if(np.sqrt(g @ g.T) < tol or runs == 4):
            return [x, f(x)[0], np.sqrt(g @ g.T)]
        a = myArmijo(x, f, p, 0.5, 0.01, 0.5)
        newx = x + a * p
        beta = (f(newx)[1] @ (f(newx)[1] - g).T) / (g @ g.T)
        p = -1 * f(newx)[1] + beta * p
        x = newx
        runs += 1

f = lambda x:myQuad(x, Q, b)
q = lambda x:myRosenbrock(x)

tol = 10^-2

[newx, nfunc, gradnorm] = myBFGS(x, f, tol)

x = np.array([12, 12])
print(nfunc)
[newx, nfunc, gradnorm] = myPR(x, q, tol)
print(nfunc)


