"""
This file contains all the necessary functions to train the JAX DPC implementation
"""

import functools
import jax
import numpy as np
import jax.numpy as jnp

# generate pytree representing all the MLP state, and return other parameters that define the MLP
def init_pol(layer_widths, parent_key, scale=0.1):

    pol_s = []
    keys = jax.random.split(parent_key, num=len(layer_widths)-1)

    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = jax.random.split(key)
        pol_s.append([
                       scale*jax.random.normal(weight_key, shape=(out_width, in_width)),
                       scale*jax.random.normal(bias_key, shape=(out_width,))
                       ]
        )

    return pol_s

def pol_inf(pol_s, s):

    # select layers which will have activation applied to them
    hidden_layers = pol_s[:-1]

    # instantiate the first layer_out, we will then iterate through network
    layer_out = s
    for w, b in hidden_layers:
        layer_out = jax.nn.relu(layer_out @ w.T + b)
    
    # We don't apply an activation func to the final output
    w_last, b_last = pol_s[-1]
    action = layer_out @ w_last.T + b_last

    # add softmax if we have multiple outputs probably
    return action # jnp.clip(action, a_min=-1, a_max=1) # jax.nn.softmax(action) # logits - logsumexp(logits)


    

    

