"""
This script demonstrates DPCs failure on a system, which IPOPT succeeds on in ipopt.py

whats more is that without the horizon being tiny at 2 the loss will become infinite and 
training will stall.
"""

seed = 0
parent_key = jax.random.PRNGKey(seed)

from jax import config
config.update("jax_enable_x64", False)
config.update('jax_platform_name', 'cpu')

import functools
import jax
import numpy as np
import jax.numpy as jnp

from utils.mlp import init_pol, pol_inf
from utils.opt import adagrad, clip_grad_norm

def f(x, u):
    # A = jnp.array([[1.2, 1.0], [0.0, 1.0]])
    # B = jnp.array([[1.0], [0.5]])
    # x_next = x @ A.T + u @ B.T
    x_1_kp1 = x[:,0:1] + x[:,0:1] ** 2 * x[:,1:2]
    x_2_kp1 = x[:,1:2] + u[:,0:1]
    x_next = jnp.hstack([x_1_kp1, x_2_kp1])
    return x_next

@functools.partial(jax.jit, static_argnums=(2,))
def b_cost(pol_s, b_s, hzn):
    loss = 0
    Q = 10.0                # state loss
    R = 0.0001              # action loss
    b = b_s.shape[0] * hzn    # number of (s,a) pairs loss is generated over
    for _ in range(hzn):
        b_a = pol_inf(pol_s, b_s)
        b_s_kp1 = f(b_s, b_a)
        loss += (R * jnp.sum(b_a**2) + Q * jnp.sum(b_s_kp1**2)) / b
        b_s = b_s_kp1
        # if jnp.isnan(loss).any():
        #     print('fin')
    return loss

def cost(pol_s, s, hzn):
    loss = 0
    Q = 10.0                # state loss
    R = 0.0001              # action loss
    b = hzn                 # number of (s,a) pairs loss is generated over
    for _ in range(hzn):
        a = pol_inf(pol_s, s)
        s_kp1 = f(s, a)
        loss += (R * jnp.sum(a**2) + Q * jnp.sum(s_kp1**2)) / b
        s = s_kp1
    return loss

def step(step, opt_s, b_s, hzn=1):
    pol_s = opt_s[0]
    loss, grads = jax.value_and_grad(b_cost)(pol_s, b_s, hzn)
    grads = clip_grad_norm(grads, max_norm=100.0)
    opt_s = opt_update(step, grads, opt_s)
    return loss, opt_s

if __name__ == "__main__":

    nx, nu = 2, 1
    lr = 1e-2
    layer_sizes = [nx, 20, 20, 20, 20, nu]
    pol_s = init_pol(layer_sizes, parent_key)

    opt_init, opt_update, get_params = adagrad(lr)
    opt_s = opt_init(pol_s) # opt_s = (pol_s, optimizer_state) i.e. all variables that change, get_params gets pol_s

    train_data = 3.0 * np.random.randn(1, 3333, 1, nx)
    num_epochs = 400 # 400
    hzn = 2

    for epoch in range(num_epochs):
        for initial_condition in train_data:  # shape = (1, 3333, 1, 2)
            b_s = jnp.squeeze(initial_condition, axis=1)  # shape = (3333, 2)
            loss, opt_s = step(epoch, opt_s, b_s, hzn=hzn)
        print(f"epoch: {epoch}, loss: {loss}")

    

    

