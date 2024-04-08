"""
After a lot of trial and error, here is a cyipopt mpc that compares to
casadi in terms of inference time - 0.002ms here vs 0.002ms casadi when
everything is jitted!
"""

import jax.numpy as jnp
from jax import grad, jacfwd, value_and_grad, jacrev, jit
from cyipopt import minimize_ipopt

def f(x, u):
    A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B = jnp.array([[0.0], [1.0]])
    x_next = A @ x + B @ u
    return x_next

def objective(z, nx=2, nu=1, N=3, Q=1.0, R=1.0):
    x = z[:N*nx].reshape(N, nx)
    u = z[N*nx:].reshape(N-1, nu)
    
    # Objective cost only considers control effort for simplicity
    cost = jnp.sum(Q * x**2) + jnp.sum(R * (u**2))
    return cost

def eq_constraints(z, nx=2, nu=1, N=3, x0=jnp.array([1., 1.])):
    x = z[:N*nx].reshape(N, nx)
    u = z[N*nx:].reshape(N-1, nu)
    
    constraints = []
    # Enforce initial condition
    constraints.append(x[0] - x0)
    
    # Enforce system dynamics as equality constraints
    for k in range(N-1):
        x_next = f(x[k], u[k])
        constraints.append(x[k+1] - x_next)
    
    return jnp.concatenate(constraints).flatten()

def ineq_constrains(ovs, nx=2, nu=1, N=3):
    # stucture of ovs: {nx_1, nx_2, ..., nx_N, nu_0, nu_1, ..., nu_N-1}
    # unpack:
    x = ovs[:nx*N]          # {nx_0, nx_1, ..., nx_N}
    u = ovs[nx*N:nx*N+nu*N] # {nu_0, nu_1, ..., nu_N}
    x = x.reshape([N, nx])  # N x nx
    u = u.reshape([N, nu])  # N x nu

    cl = [] # constraint list

    # dynamics constraints
    # x[0] is really x[1], but x[1] is first optimized state so we start from array start of 0
    # u[0] by contrast is really u[0] applied to x0
    cl.append(x[0] - 25)       # = 0
    cl.append(x[1] - 25)   # = 0
    cl.append(u[0] - 25)       # = 0
    cl.append(u[1] - 25)   # = 0

    # stacking of equality constraints
    return jnp.zeros(6) # jnp.hstack(cl) 

# Initial guess
nx, nu, N = 2, 1, 3
z_init = jnp.zeros(N*nx + (N-1)*nu)

# jit the functions
obj_jit = jit(objective)
con_eq_jit = jit(eq_constraints)
con_ineq_jit = jit(ineq_constrains)

# build the derivatives and jit them
obj_grad = jit(grad(obj_jit))  # objective gradient
obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian
con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian
# con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product
@jit
def con_eq_hessvp(x, v):
    return jnp.sum(con_eq_hess(x) * v[:, None, None], axis=0)
con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product


# Constraint definitions for IPOPT
constraints = [
    {'type': 'eq', 'fun': eq_constraints, 'jac': con_eq_jac, 'hess': con_eq_hessvp}
]

# IPOPT minimize call
result = minimize_ipopt(fun=obj_jit, x0=z_init, jac=obj_grad, hess=obj_hess, constraints=constraints, tol=1e-6, options={'disp': 5, 'maxiter': 100})

# Extract solution
sol_x = result.x[:N*nx].reshape(N, nx)
sol_u = result.x[N*nx:].reshape(N-1, nu)

result = minimize_ipopt(fun=obj_jit, x0=z_init, jac=obj_grad, hess=obj_hess, constraints=constraints, tol=1e-6, options={'disp': 5, 'maxiter': 100})

print('Optimal state trajectory:', sol_x)
print('Optimal control inputs:', sol_u)
