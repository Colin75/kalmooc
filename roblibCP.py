from numpy.linalg import inv
from numpy import *


def kalman_filter(A, C, u, y, x_pred, cov_pred, cov_b, cov_a):

    if y.size == 0:
        n = len(x_pred)
        y = eye(0, 1)
        cov_b = eye(0, 0)
        C = eye(0, n)

    # Correction
    S = C@cov_pred@C.T + cov_b
    K = cov_pred@C.T@inv(S)
    y_tilde = y - C@x_pred
    x_up = x_pred + K @ y_tilde
    cov_up = (eye(len(x_pred)) - K @ C) @ cov_pred

    # Prediction
    x_pred = A@x_up + u
    cov_pred = A@cov_up@A.T + cov_a

    return x_pred, cov_pred, x_up, cov_up


def kalman_predictor(A, u, x_hat, cov_x, cov_a):
    x_hat = A@x_hat + u
    cov_x = A@cov_x@A.T + cov_a

    return x_hat, cov_x


def kalman_smoother(A, x_pred, cov_pred, x_up, cov_up):
    N = len(x_pred)-1
    x_back = zeros_like(x_up)
    cov_back = zeros_like(cov_up)

    x_back[N] = x_up[N]
    cov_back[N] = cov_up[N]

    for k in range(N-1, -1, -1):
        J = cov_up[k]@A[k].T@inv(cov_pred[k+1])
        x_back[k] = x_up[k] + J@(x_back[k+1]-x_pred[k+1])
        cov_back[k] = cov_up[k] + J@(cov_pred[k+1]-cov_back[k+1])@J.T

    return x_back, cov_back