# This file gets same result as 2.5.py (casadi), but 50x slower at 0.076ms rather than 0.002ms
# I think this is because I had to cut objective and constraint hessians to make this work. They
# are the issue here.

import jax.numpy as jnp
from jax import grad, jacfwd, value_and_grad, jacrev, jit
from cyipopt import minimize_ipopt

def system_dynamics(x, u):
    A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B = jnp.array([[0.0], [1.0]])
    x_next = A @ x + B @ u
    return x_next

def mpc_objective(z, nx=2, nu=1, N=3, Q=1.0, R=1.0):
    x = z[:N*nx].reshape(N, nx)
    u = z[N*nx:].reshape(N-1, nu)
    
    # Objective cost only considers control effort for simplicity
    cost = jnp.sum(Q * x**2) + jnp.sum(R * (u**2))
    return cost

def dynamics_constraints(z, nx=2, nu=1, N=3, x0=jnp.array([1., 1.])):
    x = z[:N*nx].reshape(N, nx)
    u = z[N*nx:].reshape(N-1, nu)
    
    constraints = []
    # Enforce initial condition
    constraints.append(x[0] - x0)
    
    # Enforce system dynamics as equality constraints
    for k in range(N-1):
        x_next = system_dynamics(x[k], u[k])
        constraints.append(x[k+1] - x_next)
    
    return jnp.concatenate(constraints).flatten()

# Initial guess
nx, nu, N = 2, 1, 3
z_init = jnp.zeros(N*nx + (N-1)*nu)

# Objective and gradients
objective_grad = jit(grad(mpc_objective))
# obj_hess = jacrev(jacfwd(mpc_objective)) # objective hessian

constraints_jac = jit(jacfwd(dynamics_constraints))
# con_eq_hess = jacrev(jacfwd(constraints_jac)) # hessian
# con_eq_hessvp = lambda x, v: con_eq_hess(x) * v[:, None, None] # hessian vector-product

# Constraint definitions for IPOPT
constraints = {'type': 'eq', 'fun': dynamics_constraints, 'jac': constraints_jac}

# IPOPT minimize call
result = minimize_ipopt(fun=mpc_objective, x0=z_init, jac=objective_grad, constraints=[constraints], tol=1e-6, options={'disp': 5, 'maxiter': 100})

# Extract solution
sol_x = result.x[:N*nx].reshape(N, nx)
sol_u = result.x[N*nx:].reshape(N-1, nu)

print('Optimal state trajectory:', sol_x)
print('Optimal control inputs:', sol_u)
