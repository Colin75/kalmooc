from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

#
x, y = meshgrid(arange(-5, 5, 0.1), arange(-5, 5, 0.1))
z = exp(-((x-1)**2 + y**2+(x-1)*y))
fig = figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z)
fig = figure()
contour(x, y, z)

# 2D Gaussian density
n = 2
x1, x2 = meshgrid(arange(-5, 5, 0.1), arange(-5, 5, 0.1))
cov_x = eye(2)

x_bar = array([1, 0])
b = array([2, -5])

A = array([[cos(pi/6), -sin(pi/6)], [sin(pi/6), cos(pi/6)]])@array([[1, 0], [0, 3]])
cov_y = A@cov_x@A.T
y_bar = A@x_bar+b

mesh_x = np.stack((x1 - x_bar[0], x2 - x_bar[1]), axis=-1)
mesh_y = np.stack((x1 - y_bar[0], x2 - y_bar[1]), axis=-1)

qx = 1/sqrt(2*pi**n*det(cov_x))*np.exp(-0.5*mesh_x[:, :, np.newaxis, :]@inv(cov_x)@mesh_x[..., np.newaxis])
qy = 1/sqrt(2*pi**n*det(cov_y))*np.exp(-0.5*mesh_y[:, :, np.newaxis, :]@inv(cov_y)@mesh_y[..., np.newaxis])

qx = qx.squeeze()
qy = qy.squeeze()

fig = figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, qy)
fig = figure()
contour(x, y, qy)
