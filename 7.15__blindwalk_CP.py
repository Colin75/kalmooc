from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from roblibCP import *

# Correction
C = eye(2)
cov_b = array([])
y = array([])

# Prediction
u = 0
cov_a = diag([0, 0.01**2])

# init
x_hat = array([[0, 1]]).T
cov_x = diag([0, 0.02**2])

ax = init_figure(-1,12,0,2)
fig = gcf()
fig.subplots_adjust(left=29, right=30)
draw_ellipse(x_hat, cov_x, 0.99, ax, 'red')

cov_x_list = list()
det_cov = list()
for i in range(19):
    if i < 10:
        au = 1
    else :
        au = -1
    A = array([[1, au], [0, 1]])
    x_hat, cov_x = kalman_filter(A,C, u, y, x_hat, cov_x, cov_b, cov_a)
    draw_ellipse(x_hat, cov_x, 0.99, ax, [0.05 * i, 0.5, 0.5])

#     cov_x_list.append(cov_x[0][0])
#     det_cov.append(det(cov_x))
#     print(i, ' ', x_hat, '\n', cov_x, '\n',det(cov_x))
#
#
# plot(det_cov)
# plot(cov_x_list)
