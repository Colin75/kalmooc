from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from roblibCP import *

# # Model

# Estimation (Pas de mise à jour)
A = eye(2)
u = 0
cov_a = zeros([2,2])

# Innovation
C = array([[4,0],
           [10,1],
           [10,5],
           [13,5],
           [15,3]])

y = array([[5,10,11,14,17]]).T
cov_b = 9*eye(len(y))

# # Initialisation
x_hat = array([[1],[-1]])
cov_x = 4*eye(2)

if __name__ == '__main__':

    ax = init_figure(-7, 7, -7, 7)
    draw_ellipse(x_hat, cov_x, 0.9, ax, [0.8, 0.8, 0.8])
    x_hat, cov_x = kalman_filter(A,C,u,y,x_hat,cov_x,cov_b,cov_a)
    draw_ellipse(x_hat, cov_x, 0.9, ax, [0.5, 0.5, 0.5])
