"""
JAX with flax and optax
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import random
import optax

class MLP(nn.Module):
    layer_sizes: list  # List containing the sizes of each layer

    def setup(self):
        self.layers = [nn.Dense(feature_size) for feature_size in self.layer_sizes[:-1]]
        self.final_layer = nn.Dense(self.layer_sizes[-1])

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.final_layer(x)

def create_mlp_model(key, layer_sizes):
    model = MLP(layer_sizes=layer_sizes)
    params = model.init(key, jnp.ones((1, layer_sizes[0])))['params']
    return model, params

def dynamics(state, action):
    A = jnp.array([[1.2, 1.0], [0.0, 1.0]])
    B = jnp.array([[1.0], [0.5]])
    return state @ A.T + action @ B.T

def batched_cost(params, model, state):
    action = model.apply({'params': params}, state)
    next_state = dynamics(state, action)
    return (0.0001 * jnp.sum(action**2) + 10.0 * jnp.sum(next_state**2)) / state.shape[0]

@jax.jit
def clip_grad_norm(grad, max_norm):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad))[0])
    clip = lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(clip, grad)

if __name__ == "__main__":
    seed = 1
    key = random.PRNGKey(seed)
    nx, nu = 2, 1
    layer_sizes = [nx, 20, 20, 20, 20, nu]

    # Create model and optimizer
    model, params = create_mlp_model(key, layer_sizes)
    optimizer = optax.adagrad(1e-1)
    opt_state = optimizer.init(params)

    # Dataset
    train_data = 3.0 * np.random.randn(1, 3333, 1, nx)

    # Training loop
    num_epochs = 400
    for epoch in range(num_epochs):
        for initial_condition in train_data:  # shape = (1, 3333, 1, 2)
            state = jnp.squeeze(initial_condition, axis=1)  # shape = (3333, 2)
            loss, grads = jax.value_and_grad(batched_cost)(params, model, state)
            grads = clip_grad_norm(grads, max_norm=100.0)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        print(f"epoch: {epoch}, loss: {loss}")

    print('fin')