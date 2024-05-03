import jax.numpy as jnp
import jax

def clip_grad_norm(g, max_norm):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, g))[0])
    clip = lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(clip, g)

def make_schedule(scalar_or_schedule):
    if callable(scalar_or_schedule):
        return scalar_or_schedule
    elif jnp.ndim(scalar_or_schedule) == 0:
        return lambda _: scalar_or_schedule 
    else:
        raise TypeError(type(scalar_or_schedule))

# https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/optimizers.html#adagrad
def adagrad(step_size, momentum=0.9):

    step_size = make_schedule(step_size)

    def init(x0):
        g_sq = jax.tree_map(jnp.zeros_like, x0)
        m = jax.tree_map(jnp.zeros_like, x0)
        return x0, g_sq, m

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

# https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/optimizers.html#adam
def adam(step_size, b1=0.9, b2=0.999, eps=1e-8):

    step_size = make_schedule(step_size)

    def init(x0):
        m0 = jax.tree_map(jnp.zeros_like, x0)
        v0 = jax.tree_map(jnp.zeros_like, x0)
        return x0, m0, v0
    
    def update(i, g, state):
        x, m, v = state

        """
        Update m and v
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
        """
        m = jax.tree_map(lambda m_leaf, g_leaf: (1 - b1) * g_leaf + b1 * m_leaf, m, g)
        v = jax.tree_map(lambda v_leaf, g_leaf: (1 - b2) * jnp.square(g_leaf) + b2 * v_leaf, v, g)
        """
        Apply bias correction
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        """
        mhat = jax.tree_map(lambda m_leaf: m_leaf / (1 - b1 ** (i + 1)), m)
        vhat = jax.tree_map(lambda v_leaf: v_leaf / (1 - b2 ** (i + 1)), v)
        """
        Update x
        x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
        """
        x = jax.tree_map(lambda x_leaf, mhat_leaf, vhat_leaf: x_leaf - step_size(i) * mhat_leaf / (jnp.sqrt(vhat_leaf) + eps), x, mhat, vhat)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x
    
    return init, update, get_params

# https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/optimizers.html#rmsprop
def rmsprop(step_size, gamma=0.9, eps=1e-8):

    step_size = make_schedule(step_size)

    def init(x0):
        avg_sq_grad = jax.tree_map(jnp.zeros_like, x0)
        return x0, avg_sq_grad
    
    def update(i, g, state):
        x, avg_sq_grad = state
        """
        Update the avg_sq_grad using tree_map
        avg_sq_grad = avg_sq_grad * gamma + jnp.square(g) * (1. - gamma)
        """
        avg_sq_grad = jax.tree_map(lambda avg_sq_grad_leaf, g_leaf: avg_sq_grad_leaf * gamma + jnp.square(g_leaf) * (1. - gamma), avg_sq_grad, g)
        """
        Update x
        x = x - step_size(i) * g / jnp.sqrt(avg_sq_grad + eps)
        """
        x = jax.tree_map(lambda x_leaf, avg_sq_grad_leaf, g_leaf: x_leaf - step_size(i) * g_leaf / jnp.sqrt(avg_sq_grad_leaf + eps), x, avg_sq_grad, g)
        return x, avg_sq_grad
    
    def get_params(state):
        x, _ = state
        return x
    
    return init, update, get_params