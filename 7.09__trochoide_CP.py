from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
import pandas as pd
import seaborn as sns

y = array([[0.38,3.25,4.97,-0.26]]).T
ax=init_figure(-5,40,-3,8)
p=array([3,2])
t = arange(0,20,0.1)
ax.plot(p[0]*t - p[1]*sin(t) , p[0] - p[1]*cos(t),'green')
pause(1)

Cx = array([[1, sin(1)],[2, sin(2)],[3, sin(3)],[7, sin(7)]])

Cy = array([[1, cos(1)],[1, cos(2)],[1, cos(3)],[1, cos(7)]])


def trochoide(p_bar=array([[0],[0]]), cov_p=10000*eye(2)):
    cov_b = 0.01*eye(len(y))

    y_tilde = y - Cy@p_bar
    cov_y = Cy@cov_p@Cy.T+cov_b
    K = cov_p@Cy.T@inv(cov_y)
    cov_e = cov_p-K@Cy@cov_p
    p_hat = p_bar + K@y_tilde
    y_hat = Cy@p_hat
    r = norm(y - y_hat,2)

    return y_tilde, cov_y, K, cov_e, p_hat, y_hat, r


df_k = pd.DataFrame()
df_y = pd.DataFrame()

for i in range(0, 10, 1):
    for j in range(1000,10000,1000):
        p_bar = array([[i],[i]])
        cov_p = j*eye(2)

        y_tilde, cov_y, K, cov_e, p_hat, y_hat, r = trochoide(p_bar, cov_p)
        df_k.loc[i, j] = K[0][3]
        df_y.loc[i, j] = y_tilde[3]


ax.plot(p_hat[0]*t - p_hat[1]*sin(t) , p_hat[0] - p_hat[1]*cos(t),'red')

plt.axes()
sns.heatmap(df_k, cmap='RdBu')
plt.axes()
sns.heatmap(df_y, cmap='RdBu')

