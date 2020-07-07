from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from roblibCP import *


def f(x, u):
    return (array([x[3]*cos(x[4])*cos(x[2]),
                  x[3]*cos(x[4])*sin(x[2]),
                  x[3]*sin(x[4])/3,
                  u[0],
                  u[1]]))


dt = 0.1
u = array([[0,0]]).T
x = array([[0,0,pi/3,4,0.2]]).T

cov_a = diag([0, 0, 0.01*dt, 0.01*dt, 0.01*dt])

#ax=init_figure(-50,50,-50,50)

# Méthode d'Euler
x1_eul = [x[0]]
# for t in arange(0,10,dt) :
#     clear(ax)
#     draw_car(x)
#     uz = array([[0,0,dt*u[0]]]).T
#     x = x + dt*f(x,u) + mvnrnd1(cov_a)
#     x1_eul.append(x[0])
# pause(1)

# # #  Méthode Kalman

# # Model : correction (aucune)

# # Model : prediction
dt = 0.1

x = array([[0,0,pi/3,4,0.2]]).T
z_hat = array([x[0], x[1], x[3]])
cov_z = zeros([3,3])

cov_ax = dt*diag([0, 0, 0.01, 0.01, 0.01])
cov_az = dt*diag([0.01, 0.01, 0.01])

# # Filtrage
ax = init_figure(-50,50,-50,50)
x_res = [x[0][0]]
cov_x_res = [cov_z[0,0]]

for t in arange(0,50,dt):
    ux = array([[0, 0]]).T
    x = x + dt * f(x, ux) + mvnrnd1(cov_ax)

    uz = dt * array([[0, 0, ux[0][0]]]).T

    y = array([])
    C = eye(1)
    cov_b = array([])

    A = array([[1, 0, dt * cos(x[4][0]) * cos(x[2][0])],
               [0, 1, dt * cos(x[4][0]) * sin(x[2][0])],
               [0, 0, 1]])

    z_hat, cov_z = kalman_filter(A, C, uz, y, z_hat, cov_z, cov_b, cov_az)
    print(z_hat)

    if t in range(100):
        draw_car(x)
        draw_ellipse(z_hat[:2], cov_z[:2, :2], 0.9, ax, 'black')

    # clear(ax)
    # draw_car(x)
    # draw_ellipse(z_hat[:2], cov_z[:2, :2], 0.9, ax, 'black')

    x_res.append(x[0])
    cov_x_res.append(cov_z[0,0])

