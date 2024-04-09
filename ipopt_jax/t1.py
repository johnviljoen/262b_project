"""
attempt to integrate dynamics type equality constraints
"""

from jax import config

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')

import jax.numpy as np
from jax import jit, grad, jacfwd, jacrev, numpy as jnp
from cyipopt import minimize_ipopt

def objective(ovs, nx=2, nu=1):
    x, u = ovs[:nx], ovs[nx:]
    return np.sum(x**2) + np.sum(u**2)

def dynamics(x, u):
    A = jnp.array([[1.0, 1.0],[0.0,1.0]])
    B = jnp.array([[0.0],[1.0]])
    x_kp1 = A @ x + B @ u
    return x_kp1

def eq_constraints(ovs, hzn=2, nx=2, nu=1):


    return np.sum(x_list**2, axis=1) - 40

# nonlinear inequality constraints only - bounds are defined later for box constraints
def ineq_constrains(ovs):
    return np.prod(ovs) - 25

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
con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product

# constraints
cons = [
    {'type': 'eq', 'fun': con_eq_jit, 'jac': con_eq_jac, 'hess': con_eq_hessvp},
    {'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp}
 ]

# starting point
x0 = np.array([1.0, 5.0, 5.0])

# variable bounds: 1 <= x[i] <= 5
bnds = [(1, 5) for _ in range(x0.size)]

# executing the solver
res = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=x0, bounds=bnds,
                  constraints=cons, options={'disp': 5})

