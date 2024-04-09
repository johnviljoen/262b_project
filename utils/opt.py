import jax.numpy as jnp
import jax

def clip_grad_norm(g, max_norm):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, g))[0])
    clip = lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(clip, g)

def adagrad(step_size, momentum=0.9):

    def make_schedule(scalar_or_schedule):
        if callable(scalar_or_schedule):
            return scalar_or_schedule
        elif jnp.ndim(scalar_or_schedule) == 0:
            return lambda _: scalar_or_schedule 
        else:
            raise TypeError(type(scalar_or_schedule))

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
