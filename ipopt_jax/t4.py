"""
This script will implement a basic MPC using cyipopt in a non-batched approach
works for only jacobian inputs, but slower than casadi...
"""

from jax import config

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev
from cyipopt import minimize_ipopt

def f(x, u):
    A = jnp.array([[1.0, 1.0],[0.0,1.0]])
    B = jnp.array([[0.0],[1.0]])
    x_kp1 = A @ x + B @ u
    return x_kp1

mpc_params = {
    "nx": 2,
    "nu": 1,
    "N": 2,
    "x0": jnp.array([1., 1.])
}


def objective(ovs, mpc_params=mpc_params): # a function of the optimization variables
    # stucture of ovs: {nx_1, nx_2, ..., nx_N, nu_0, nu_1, ..., nu_N-1}
    # unpack:
    nx, nu, N = mpc_params["nx"], mpc_params["nu"], mpc_params["N"]
    x = ovs[:nx*N]          # {nx_0, nx_1, ..., nx_N}
    u = ovs[nx*N:nx*N+nu*N] # {nu_0, nu_1, ..., nu_N}
    x = x.reshape([N, nx])  # N x nx
    u = u.reshape([N, nu])  # N x nu

    Q = jnp.array([1, 1])    # np.eye(nx)
    R = jnp.array([1])       # np.eye(nu)

    # efficient version of x.T @ Q @ x + u.T @ R @ u across the sequence of x,u pairs
    return jnp.sum(x ** 2 * Q + u ** 2 * R)


# test reshaping in obj func:
ovs = jnp.array([1,2,3,4,5,6])
objective(ovs, mpc_params)

def eq_constraints(ovs, mpc_params=mpc_params):
    # stucture of ovs: {nx_1, nx_2, ..., nx_N, nu_0, nu_1, ..., nu_N-1}
    # unpack:
    nx, nu, N, x0 = mpc_params["nx"], mpc_params["nu"], mpc_params["N"], mpc_params["x0"]
    x = ovs[:nx*N]          # {nx_0, nx_1, ..., nx_N}
    u = ovs[nx*N:nx*N+nu*N] # {nu_0, nu_1, ..., nu_N}
    x = x.reshape([N, nx])  # N x nx
    u = u.reshape([N, nu])  # N x nu

    cl = [] # constraint list

    # initial conditions
    # cl.append(x[0] - x0)               # = 0

    # dynamics constraints
    # x[0] is really x[1], but x[1] is first optimized state so we start from array start of 0
    # u[0] by contrast is really u[0] applied to x0
    cl.append(x[0] - f(x0, u[0]))       # = 0
    cl.append(x[1] - f(x[0], u[1]))   # = 0

    # stacking of equality constraints
    return jnp.hstack(cl)

# test equality constraints
eq_constraints(ovs, mpc_params)

def ineq_constrains(ovs, mpc_params=mpc_params):
    # stucture of ovs: {nx_1, nx_2, ..., nx_N, nu_0, nu_1, ..., nu_N-1}
    # unpack:
    nx, nu, N, x0 = mpc_params["nx"], mpc_params["nu"], mpc_params["N"], mpc_params["x0"]
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
con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product
def con_eq_hessvp(x, v):
    return jnp.sum(con_eq_hess(x) * v[:, None, None], axis=0)
con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product

# constraints
cons = [
    {'type': 'eq', 'fun': con_eq_jit, 'jac': con_eq_jac, 'hess': con_eq_hessvp},
    # {'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp}
 ]

# starting point
# x0 = jnp.array([1.0, 5.0])
x0 = jnp.array([1.,2.,3.,4.,5.,6.])
x0 = jnp.zeros(6)

# variable bounds: 1 <= x[i] <= 5
bnds = [(-10, 10) for _ in range(x0.size)]

"""TESTING START"""
# test = con_eq_hess(x0)
# test2 = obj_hess(x0)
"""TESTING END"""

# executing the solver
res = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=x0, bounds=bnds,
                  constraints=cons, tol=1e-6, options={'disp': 5})

nx, nu, N, x0 = mpc_params["nx"], mpc_params["nu"], mpc_params["N"], mpc_params["x0"]
sol_x = res.x[:nx*N]
sol_u = res.x[nx*N:]

print(sol_x)
print(sol_u)

print('f')