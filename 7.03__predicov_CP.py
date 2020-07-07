from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from scipy.stats import norm
from numpy.random import multivariate_normal as mnorm

N = 1000
X1 = randn(2, N)
#print('X=', X1)

xbar = zeros([2, 1])
cov_x = eye(2)

x2_bar = array([[1], [2]])
cov_x2 = array([[4, 3], [3, 3]])
X2 = x2_bar + sqrtm(cov_x2) @ X1

#%% Ellipse confidence
ax = init_figure(-10, 10, -10, 10)
draw_ellipse(x2_bar, cov_x2, 0.9, ax, [1, 0.8, 0.8])
draw_ellipse(x2_bar, cov_x2, 0.99, ax, [1, 0.8, 0.8])
draw_ellipse(x2_bar, cov_x2, 0.999, ax, [1, 0.8, 0.8])
scatter(X2[0], X2[1], s=2)

pause(1)

x2_bar_emp = sum(X2, axis=1, keepdims=True)/N

# (n?,k),(k,m?)->(n?,m?)

vars_x2 = sum((X2-x2_bar)**2, axis=1)/N
cov_x2_xy = sum((X2[0]-x2_bar[0])@(X2[1]-x2_bar[1]).T)/N
cov_x2_emp = diag(vars_x2)
cov_x2_emp[0][:] = cov_x2_xy
cov_x2_emp[:][0] = cov_x2_xy

#%% Euler discretization


def euler_state(X, t, dt):
    global Ad, B
    noise = norm.rvs(0, dt, size=2)
    ud = dt*B*sin(t)
    X_new = Ad @ X + ud + noise.reshape(2, 1)
#    X_new = X + dt*(A@X + (B*sin(t))) + noise.reshape(2, 1)

    return X_new

#%% Kalman prediction


def kal_state(X_bar, cov_X, t, dt):

    global Ad, B

    noise = norm.rvs(0, dt, size=2)
    cov_noise = np.cov(noise)

    ud = dt*B*sin(t)
    X_bar = Ad@X_bar + ud
    cov_X = Ad@cov_X@Ad.T + cov_noise

    return X_bar, cov_X

#%% Pose simulation


A = array([[0,1],[-1,0]])
B = array([[2], [3]])

step = 6
dt = 0.01
Ad = eye(2)+dt*A

X_euler = X2
X_bar = X2.mean(axis=1, keepdims=True)
cov_X = np.cov(X2)

ax = init_figure(-13, 13, -13, 13)
color = iter(cm.rainbow(np.linspace(0,1,step+1)))

for t in arange(0, step+dt, dt):
    X_bar, cov_X = kal_state(X_bar, cov_X, t, dt)
    X_euler = euler_state(X_euler, t, dt)
    if t in range(7):
        c = next(color)
        draw_ellipse(X_bar, cov_X, 0.9, ax, c)
        scatter(X_euler[0], X_euler[1], color=c, label=f'{t}', s=1)
    legend()

