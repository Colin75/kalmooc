from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from roblibCP import *

x = array([[0,0]]).T
cov_x = 50*eye(2)
ax=init_figure(-100,100,-100,100)
#draw_ellipse(x,cov_x,0.99,ax,[1,0.8,0.8])
pause(1)


# Prediction (x_k -> x_{k+1})
A = eye(2)        # Sans transformation
u = 0             # Sans entrée
cov_a = zeros([2,2])  # "Donc" sans incertitude

# Correction (y_k -> x_{k+1})
C = array([[2,3],
           [3,2],
           [1,-1]])
y = array([[8], [7], [0]])  # observations
cov_b = diag([1, 4, 4])     # variance du bruits

# # Résolution en 3 étapes

# Initialisation
x_pred = array([[0],[0]])
cov_pred = 1000*eye(2)

lim = 250
ax = init_figure(-lim,lim,-lim,lim)
print(f'step 0\n{x_pred}, \n{cov_pred} \n')
draw_ellipse(x_pred, cov_pred, 0.90, ax, [1, 0.8, 0.8])

for k in range(3):
    x_pred, cov_pred = kalman_filter(A, C[None, k], u, y[None, k], x_pred, cov_pred, cov_b[k][k], cov_a)
    print(f'step {k + 1}\n{x_pred}, \n{cov_pred} \n')
    draw_ellipse(x_pred, cov_pred, 0.90, ax, [1-(k+1)*0.2, 0.8, 0.8])


# # Résolution en 1 étape

# Initialisation
x_pred = array([[0],[0]])
cov_pred = 1000*eye(2)

x_pred, cov_pred = kalman_filter(A, C, u, y, x_pred, cov_pred, cov_b, cov_a)
print(f'{x_pred}, \n{cov_pred} \n')
draw_ellipse(x_pred, cov_pred, 0.90, ax, [0.5, 0.3, 0.3])


