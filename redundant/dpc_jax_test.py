import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax
from jax import jit, vmap, pmap, grad, value_and_grad


# generate pytree representing all the MLP state, and return other parameters that define the MLP
def init_mlp(layer_widths, parent_key, scale=0.01):

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


def mlp_predict(mlp_state, observation):

    # select layers which will have activation applied to them
    hidden_layers = mlp_state[:-1]

    # instantiate the first layer_out, we will then iterate through network
    layer_out = observation
    for w, b in hidden_layers:
        layer_out = jax.nn.relu(w @ layer_out + b)
    
    # We don't apply an activation func to the final output
    w_last, b_last = mlp_state[-1]
    action = w_last @ layer_out + b_last

    # - logsumexp(logits) implicitly applies softmax
    return action # jnp.clip(action, a_min=-1, a_max=1) # jax.nn.softmax(action) # logits - logsumexp(logits)


# 1D, we will vmap later - make sure to get mean across batch!
def batched_cost(mlp_state, state):
    action = batched_MLP_predict(mlp_state, state)
    next_state = dynamics(state, action)
    return ((0.0001 * action.T @ action + 10.0 * jnp.einsum('ij,ij->', next_state, next_state))/state.shape[0])[0,0]

def dynamics(state, action):
    A = jnp.array([[1.2, 1.0],
                    [0.0, 1.0]])
    B = jnp.array([[1.0],
                    [0.5]])
    return state @ A.T + action @ B.T


# implement SGD update
@jit
def update(mlp_params, inputs, targets, lr=0.01):
    loss, grads = value_and_grad(vmap(cost))(mlp_params, inputs, targets)
    return loss, jax.tree_map(lambda p, g: p - lr*g, mlp_params, grads)

# reference adagrad implementation with constant step size:
# https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/optimizers.html#adagrad
def adagrad(step_size, momentum=0.9):

  def init(x0):
    g_sq = jnp.zeros_like(x0)
    m = jnp.zeros_like(x0)
    return x0, g_sq, m

  def update(i, g, state):
    x, g_sq, m = state
    g_sq += jnp.square(g)
    g_sq_inv_sqrt = jnp.where(g_sq > 0, 1. / jnp.sqrt(g_sq), 0.0)
    m = (1. - momentum) * (g * g_sq_inv_sqrt) + momentum * m
    x = x - step_size(i) * m
    return x, g_sq, m

  def get_params(state):
    x, _, _ = state
    return x

  return init, update, get_params


def clip_grad_norm(grad, max_norm):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad))[0])
    clip = lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(clip, grad)


if __name__ == "__main__":
    seed = 0
    key = jax.random.PRNGKey(seed)
    nx = 2
    nu = 1
    mlp_state = init_mlp([nx, 20, 20, 20, 20, nu], key)

    # tests
    inp = np.random.randn(2)
    print(inp.shape)

    prediction = mlp_predict(mlp_state, inp)
    print(prediction)

    batched_MLP_predict = vmap(mlp_predict, in_axes=(None, 0))
    batched_inp = np.random.randn(16, nx)
    predictions = batched_MLP_predict(mlp_state, batched_inp)
    print(predictions)

    # dataset
    train_data = 3.*np.random.randn(1, 3333, 1, nx)

    horizon = 2
    num_epochs = 400
    lr = 1e-2
    # opt_init, opt_update, get_params = adagrad(lr)
    # opt_state = opt_init(mlp_state)

    for epoch in range(num_epochs):
        for initial_condition in train_data: # (state.shape = 3333x1x2)
            state = initial_condition[:,0,:]
            # for timestep in range(horizon):
            action = batched_MLP_predict(mlp_state, state)
            next_state = dynamics(state, action)
            loss, grads = jit(value_and_grad(batched_cost))(mlp_state, state)
            grads = clip_grad_norm(grads, max_norm=100.0)
            # loss, grads = jit(value_and_grad(batched_cost))(get_params(opt_state))
            # opt_state = opt_update(epoch, grads, opt_state)
            mlp_state = jax.tree_map(lambda p, g: p - lr*g, mlp_state, grads)
        print(f"epoch: {epoch}, loss: {loss}")
    print('fin')



