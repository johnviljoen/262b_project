import jax
import numpy as np

# Function to reconstruct the original leaves from the concatenated vector
def reconstruct_pytree(vector, shapes_and_dtypes, treedef):
    offset = 0
    new_leaves = []
    for shape, dtype in shapes_and_dtypes:
        size = np.prod(shape)
        new_leaves.append(vector[offset:offset+size].reshape(shape).astype(dtype))
        offset += size
    new_pytree = jax.tree.unflatten(treedef, new_leaves)
    # assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: jnp.allclose(x, y), original_pytree, new_pytree))
    return new_pytree