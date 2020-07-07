from scipy.stats import uniform
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def monte_carlo(y_target, tol=1e-1, iter=1000):
    """
    tol : float, default=1e-3
    Tolerance for stopping criteria.
    """

    sol = list()
    out = list()

    a = uniform.rvs(loc=0, scale=2, size=iter, random_state=12)
    b = uniform.rvs(loc=0, scale=2, size=iter, random_state=46)

    for i in range(iter):
        y = solver(a[i], b[i])

        if norm(y - y_target, ord=np.inf) < tol:
            print(f'CV reached : a = {a[i]:0.2f}, b = {b[i]:0.2f}')
            sol.append([a,b])
        else:
            out.append([a,b])
    return sol, out


def solver(a, b):
    u = 1
    x = np.array([0, 0])
    y = np.zeros([6])

    for k in range(6):
        y[k] = np.array([1, 1])@x
        x = np.array([[1, 0], [a, 0.9]])@x + np.array([b, 1 - b]) * u

    return y


def solver2(a, b):
    u = 1
    x = np.zeros([7, 2])
    x[0] = np.array([0, 0])
    y = np.zeros([6])

    for k in range(6):
        x[k+1] = np.array([[1, 0], [a, 0.9]])@x[k] + np.array([b, 1 - b]) * u
        y[k] = np.array([1, 1]) @ x[k]

    return y


y_target = np.array([0, 1, 2.65, 4.885, 7.646, 10.882])
sol, out = monte_carlo(y_target, tol=1e-2, iter=10000)
