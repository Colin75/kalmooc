from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from roblibCP import *

# # Model
# Prediction
A = array([[[0.5, 0], [0, 1]],
           [[1, -1], [1, 1]],
           [[1, -1], [1, 1]]])
u = array([[[8], [16]],
           [[-6], [-18]],
           [[32], [-8]]])
cov_a = eye(2)

# Observation
y = array([[7],
           [30],
           [6]])
C = array([[1., 1.]])
cov_b = 1


if __name__ == '__main__':

    ax = init_figure(-100, 100, -100, 100)
    x_hat = array([[0], [0]])
    cov_x = 100*eye(2)

    for k in range(3):
        x_hat, cov_x = kalman_filter(A[k], C, u[k], y[k], x_hat, cov_x, cov_b, cov_a)
        draw_ellipse(x_hat, cov_x, 0.9, ax, [1-(k+1)*0.2, 0.8, 0.8])
