"""
JAX with optax, no flax, works great!!! - also with multiple step rollouts now! - JITted!
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import optax
import functools

from jax import config
config.update("jax_enable_x64", True)

# generate pytree representing all the MLP state, and return other parameters that define the MLP
def init_pol(layer_widths, parent_key, scale=0.1):

    mlp_state = []
    keys = jax.random.split(parent_key, num=len(layer_widths)-1)

    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = jax.random.split(key)
        mlp_state.append([
                       scale*jax.random.normal(weight_key, shape=(out_width, in_width)),
                       scale*jax.random.normal(bias_key, shape=(out_width,))
                       ]
        )

    return mlp_state

def pol_inf(mlp_state, observation):

    # select layers which will have activation applied to them
    hidden_layers = mlp_state[:-1]

    # instantiate the first layer_out, we will then iterate through network
    layer_out = observation
    for w, b in hidden_layers:
        layer_out = jax.nn.relu(w @ layer_out.T + b[:,None]).T
    
    # We don't apply an activation func to the final output
    w_last, b_last = mlp_state[-1]
    action = (w_last @ layer_out.T + b_last[:,None]).T

    # - logsumexp(logits) implicitly applies softmax
    # add softmax if we have multiple outputs probably
    return action # jnp.clip(action, a_min=-1, a_max=1) # jax.nn.softmax(action) # logits - logsumexp(logits)

def f(s, a):
    A = jnp.array([[1.2, 1.0], [0.0, 1.0]])
    B = jnp.array([[1.0], [0.5]])
    return s @ A.T + a @ B.T

@functools.partial(jax.jit, static_argnums=(2,))
def b_cost(pol_s, s, hzn):
    loss = 0
    Q = 10.0                # state loss
    R = 0.0001              # action loss
    b = s.shape[0] * hzn    # number of (s,a) pairs loss is generated over
    for _ in range(hzn):
        a = pol_inf(pol_s, s)
        s_kp1 = f(s, a)
        loss += (R * jnp.sum(a**2) + Q * jnp.sum(s_kp1**2)) / b
        s = s_kp1
    return loss

def clip_grad_norm(g, max_norm):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, g))[0])
    clip = lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(clip, g)

seed = 0
key = random.PRNGKey(seed)
nx, nu = 2, 1
lr = 1e-2
layer_sizes = [nx, 20, 20, 20, 20, nu]
pol_s = init_pol(layer_sizes, key)
opt = optax.adam(lr)
opt_s = opt.init(pol_s)

train_data = 3.0 * np.random.randn(1, 3333, 1, nx)
num_epochs = 400
hzn = 10
for epoch in range(num_epochs):
    for initial_condition in train_data:  # shape = (1, 3333, 1, 2)
        s = jnp.squeeze(initial_condition, axis=1)  # shape = (3333, 2)
        loss, grads = jax.value_and_grad(b_cost)(pol_s, s, hzn)
        grads = clip_grad_norm(grads, max_norm=100.0)
        updates, opt_s = opt.update(grads, opt_s)
        pol_s = optax.apply_updates(pol_s, updates)
    print(f"epoch: {epoch}, loss: {loss}")