"""
This file we will run an experiment which demonstrates DPCs failure
and IPOPTs success.
"""

import jax.numpy as jnp
from jax import grad, jacfwd, value_and_grad, jacrev, jit
from cyipopt import minimize_ipopt

def f(x, u):
    # A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    # B = jnp.array([[0.0], [1.0]])
    # x_next = A @ x + B @ u
    x_1_kp1 = x[0] + x[0]**2 * x[1]
    x_2_kp1 = x[1] + u
    x_next = jnp.hstack([x_1_kp1, x_2_kp1])
    return x_next

def l(z, nx=2, nu=1, N=3, Q=1.0, R=1.0):
    x = z[:N*nx].reshape(N, nx)
    u = z[N*nx:].reshape(N-1, nu)
    
    # Objective cost only considers control effort for simplicity
    cost = jnp.sum(Q * x**2) + jnp.sum(R * (u**2))
    return cost

def h(z, nx=2, nu=1, N=3, x0=jnp.array([1., 1.])):
    x = z[:N*nx].reshape(N, nx)
    u = z[N*nx:].reshape(N-1, nu)
    
    cl = []
    # Enforce initial condition
    cl.append(x[0] - x0)
    
    # Enforce system dynamics as equality constraints
    for k in range(N-1):
        x_next = f(x[k], u[k])
        cl.append(x[k+1] - x_next)
    
    return jnp.concatenate(cl).flatten()

def g(z, nx=2, nu=1, N=3):
    x = z[:N*nx].reshape(N, nx)
    u = z[N*nx:].reshape(N-1, nu)

    cl = [] # constraint list

    # random inequalities - can be nonlinear ofc
    cl.append(x[0] + 25)       # = 0
    cl.append(x[1] + 25)   # = 0
    cl.append(u[0] + 25)       # = 0
    cl.append(u[1] + 25)   # = 0

    # stacking of equality constraints
    return jnp.hstack(cl) 

# Initial guess
nx, nu, N = 2, 1, 3
z_init = jnp.zeros(N*nx + (N-1)*nu)

# jit the functions
obj_jit = jit(l)
con_eq_jit = jit(h)
con_ineq_jit = jit(g)

# build the derivatives and jit them
obj_grad = jit(grad(obj_jit))  # objective gradient
obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian
con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian
con_eq_hessvp = jit(lambda x, v: jnp.sum(con_eq_hess(x) * v[:, None, None], axis=0))
con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
con_ineq_hessvp = jit(lambda x, v: jnp.sum(con_ineq_hess(x) * v[:, None, None], axis=0))

# Constraint definitions for IPOPT
constraints = [
    {'type': 'eq', 'fun': h, 'jac': con_eq_jac, 'hess': con_eq_hessvp},
    {'type': 'ineq', 'fun': g, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp}
]

# IPOPT minimize call
result = minimize_ipopt(fun=obj_jit, x0=z_init, jac=obj_grad, hess=obj_hess, constraints=constraints, tol=1e-6, options={'disp': 5, 'maxiter': 100})

# Extract solution
sol_x = result.x[:N*nx].reshape(N, nx)
sol_u = result.x[N*nx:].reshape(N-1, nu)

result = minimize_ipopt(fun=obj_jit, x0=z_init, jac=obj_grad, hess=obj_hess, constraints=constraints, tol=1e-6, options={'disp': 5, 'maxiter': 100})

print('Optimal state trajectory:', sol_x)
print('Optimal control inputs:', sol_u)
