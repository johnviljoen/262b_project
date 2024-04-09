"""Test 1
xxx_s: references the "state" of something
a: action
b_xxx: refernces a "batched" version of something
"""

import jax
import numpy as np
import jax.numpy as jnp
from mlp import init_mlp
from mlp import mlp_inf


def cost(mlp_s, s):
    Q, R = jnp.eye(2), jnp.eye(1)
    a = mlp_inf(mlp_s, s)
    s_kp1 = f(s, a)
    return 10.0 * s_kp1.T @ Q @ s_kp1 + 0.0001 * a.T @ R @ a


def f(s, a):
    A = jnp.array([[1.2, 1.0], [0.0, 1.0]])
    B = jnp.array([[1.0], [0.5]])
    return A @ s + B @ a


seed = 0
key = jax.random.PRNGKey(seed)
nx = 2
nu = 1
mlp_state = init_mlp([nx, 20, 20, 20, 20, nu], key)
train_data = 3.*jax.random.normal(key, (3333, 1, nx))
b_mlp_inf = jax.vmap(mlp_inf, in_axes=(None, 0)) # to test
test1 = b_mlp_inf(mlp_state, train_data[:,0,:])
test2 = mlp_inf(mlp_state, train_data[0,0,:])
test3 = mlp_inf(mlp_state, train_data[1,0,:])
print('fin')