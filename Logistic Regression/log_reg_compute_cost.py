import numpy as np
# Linear Regression


# z = w[1]*x[1]+w[2]*x[3]+b

# g = 1/(1+e**(-z))

# Cost Function

# Without Regularization
def log_cost_function(x,y,w,b):
    m=x.shape[0]

    cost=0  
    for i in range(m):

        # for Single feature
        z_i = w*x[i]+b

        # for multiple feature
        z_i = np.dot(w*x[i])+b

        g_i = 1/(1+np.exp(-z_i))

        cost = -y[i]*np.log(g_i)-(1-y[i]*np.log(1-y[i]))

        cost = cost + cost

    total_cost = cost/m

    return total_cost

# Regularized
def reg_log_cost_function(x,y,w,b,Lam_var):
    cost=0
    m=x.shape[0]
    n=len(b)

    for i in range(m):

        # for Single feature
        z_i = w*x[i]+b

        # for multiple feature
        z_i = np.dot(w*x[i])+b

        g_i = 1/(1+np.exp(-z_i))

        cost = -y[i]*np.log(g_i)-(1-y[i]*np.log(1-y[i]))

        cost = cost + cost
         
    cost=cost/m

    reg_cost=0
    for i in range(n):
       reg_cost += (Lam_var)*(w[n])**2
    reg_cost = reg_cost/2*m

    total_cost = cost + reg_cost

    return total_cost

# Gradient Function




