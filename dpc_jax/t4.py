"""
FULL DPC w/ JAX w/out optax, flax, works great!!!
"""

import jax
import jax.numpy as jnp
import numpy as np
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

from typing import Callable

Step = int
Schedule = Callable[[Step], float]

def constant(step_size) -> Schedule:
  def schedule(i):
    return step_size
  return schedule

def make_schedule(scalar_or_schedule: float | Schedule) -> Schedule:
  if callable(scalar_or_schedule):
    return scalar_or_schedule
  elif jnp.ndim(scalar_or_schedule) == 0:
    return constant(scalar_or_schedule)
  else:
    raise TypeError(type(scalar_or_schedule))

def make_schedule(scalar_or_schedule):
    if callable(scalar_or_schedule):
        return scalar_or_schedule
    elif jnp.ndim(scalar_or_schedule) == 0:
        return lambda _: scalar_or_schedule 
    else:
        raise TypeError(type(scalar_or_schedule))

def adagrad(step_size, momentum=0.9):

    """ Example Usage:
    opt_init, opt_update, get_params = optimizers.adagrad(learning_rate)
    opt_state = opt_init(params)

    def step(step, opt_state):
        value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    for i in range(num_steps):
        value, opt_state = step(i, opt_state)
    """

    step_size = make_schedule(step_size)

    def init(x0):
        g_sq = jax.tree_map(jnp.zeros_like, x0)
        m = jax.tree_map(jnp.zeros_like, x0)
        return x0, g_sq, m

    # def update(i, g, state):
    #     x, g_sq, m = state
    #     g_sq += jnp.square(g)
    #     g_sq_inv_sqrt = jnp.where(g_sq > 0, 1. / jnp.sqrt(g_sq), 0.0)
    #     m = (1. - momentum) * (g * g_sq_inv_sqrt) + momentum * m
    #     x = x - step_size(i) * m
    #     return x, g_sq, m
    
    def update(i, g, state):
        x, g_sq, m = state
        # Update g_sq
        g_sq = jax.tree_map(lambda g_sq_leaf, g_leaf: g_sq_leaf + jnp.square(g_leaf), g_sq, g)
        # Compute g_sq_inv_sqrt and m
        g_sq_inv_sqrt = jax.tree_map(lambda g_sq_leaf: jnp.where(g_sq_leaf > 0, 1. / jnp.sqrt(g_sq_leaf), 0.0), g_sq)
        m = jax.tree_map(lambda m_leaf, g_leaf, g_sq_inv_sqrt_leaf: (1. - momentum) * (g_leaf * g_sq_inv_sqrt_leaf) + momentum * m_leaf, m, g, g_sq_inv_sqrt)
        # Update x
        x = jax.tree_map(lambda x_leaf, m_leaf: x_leaf - step_size(i) * m_leaf, x, m)
        return x, g_sq, m

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params

seed = 0
key = jax.random.PRNGKey(seed)
nx, nu = 2, 1
lr = 1e-2
layer_sizes = [nx, 20, 20, 20, 20, nu]
pol_s = init_pol(layer_sizes, key)

opt_init, opt_update, get_params = adagrad(lr)
opt_s = opt_init(pol_s)

def step(step, opt_s):
    pol_s = opt_s[0]
    loss, grads = jax.value_and_grad(b_cost)(pol_s, s, hzn)
    grads = clip_grad_norm(grads, max_norm=100.0)
    opt_s = opt_update(step, grads, opt_s)
    return loss, opt_s

train_data = 3.0 * np.random.randn(1, 3333, 1, nx)
num_epochs = 400
hzn = 10
for epoch in range(num_epochs):
    for initial_condition in train_data:  # shape = (1, 3333, 1, 2)
        s = jnp.squeeze(initial_condition, axis=1)  # shape = (3333, 2)
        loss, opt_s = step(epoch, opt_s)

        # pol_s = optax.apply_updates(pol_s, updates)
    print(f"epoch: {epoch}, loss: {loss}")