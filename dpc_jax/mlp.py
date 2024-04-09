import jax
import numpy as np
import jax.numpy as jnp

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


def mlp_inf(mlp_state, observation):

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


