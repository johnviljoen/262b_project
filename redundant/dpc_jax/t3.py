"""
JAX with optax, no flax, works great!!!
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import optax

from jax import config
config.update("jax_enable_x64", True)
# generate pytree representing all the MLP state, and return other parameters that define the MLP
def init_mlp(layer_widths, parent_key, scale=0.1):

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

def mlp_inf(mlp_state, observation):

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

def dynamics(state, action):
    A = jnp.array([[1.2, 1.0], [0.0, 1.0]])
    B = jnp.array([[1.0], [0.5]])
    return state @ A.T + action @ B.T

def batched_cost(params, state):
    action = mlp_inf(params, state)
    next_state = dynamics(state, action)
    return (0.0001 * jnp.sum(action**2) + 10.0 * jnp.sum(next_state**2)) / state.shape[0]

@jax.jit
def clip_grad_norm(grad, max_norm):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad))[0])
    clip = lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(clip, grad)

if __name__ == "__main__":
    seed = 0
    key = random.PRNGKey(seed)
    nx, nu = 2, 1
    layer_sizes = [nx, 20, 20, 20, 20, nu]

    # Create model and optimizer
    params = init_mlp(layer_sizes, key)
    # model, params = create_mlp_model(key, layer_sizes)
    optimizer = optax.adagrad(1e-1)
    opt_state = optimizer.init(params)

    # Dataset
    train_data = 3.0 * np.random.randn(1, 3333, 1, nx)

    # Training loop
    num_epochs = 400
    for epoch in range(num_epochs):
        for initial_condition in train_data:  # shape = (1, 3333, 1, 2)
            state = jnp.squeeze(initial_condition, axis=1)  # shape = (3333, 2)
            loss, grads = jax.value_and_grad(batched_cost)(params, state)
            grads = clip_grad_norm(grads, max_norm=100.0)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        print(f"epoch: {epoch}, loss: {loss}")

    print('fin')