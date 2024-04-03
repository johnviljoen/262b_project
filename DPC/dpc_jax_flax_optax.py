import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
import optax

key = random.PRNGKey(1)

def dynamics(state, action):
    A = jnp.array([[1.2, 1.0],
                   [0.0, 1.0]])
    B = jnp.array([[1.0],
                   [0.5]])
    return state @ A.T + action @ B.T

def cost(state, action):
    state_loss = jnp.einsum('ijk,ilk->jl', state, state)
    action_loss = jnp.einsum('ijk,ilk->jl', action, action)
    return (0.0001 * action_loss + 10.0 * state_loss)/state.shape[0]

class MLP(nn.Module):
    nx: int
    nu: int
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        # Define hsizes directly here
        hsizes = [20, 20, 20, 20]
        for hsize in hsizes:
            x = nn.Dense(features=hsize, use_bias=self.bias)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.nu, use_bias=self.bias)(x)
        return x

def create_optimizer(params, learning_rate=0.01, clip=100.0):
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip),
        optax.adam(learning_rate)
    )
    return optax.apply_updates(params, optimizer.init(params))

def train_step(state, policy, optimizer, params):
    def loss_fn(params):
        action = policy.apply(params, state)
        next_state = dynamics(state, action)
        return cost(next_state, action)
    
    grads = jax.grad(loss_fn)(params)
    return optimizer.update(grads, params)

def train(policy, train_data, epochs=1):
    params = policy.init(key, train_data[0])
    optimizer = create_optimizer(params)
    for epoch in range(epochs):
        total_loss = 0.0
        for state in train_data:
            params, opt_state = train_step(state, policy, optimizer, params)
            total_loss += loss_fn(params).detach()
        print(f'Epoch {epoch}, Loss: {total_loss / len(train_data)}')

if __name__ == "__main__":
    nx = 2
    nu = 1
    train_data = 3. * random.normal(key, (1, 3333, 1, nx))
    policy = MLP(nx, nu)

    # test batched jacobian
    dummy_input = jnp.ones([10, 2])
    params = policy.init(key, dummy_input)
    model_apply = lambda x: policy.apply(params, x)

    # batching occurs in 0th dimension when in_axes specified like this
    model_apply_batch = jax.vmap(model_apply, in_axes=0)

    # Directly compute the Jacobian for a single input to test
    jacobian_func = jax.jacfwd(model_apply)
    try:
        # Test with a single input to ensure it works
        jacobian_single = jacobian_func(dummy_input[0])
        print("Single Jacobian calculation successful.")
    except Exception as e:
        print("Error computing single Jacobian:", e)

    # If the above succeeds without errors, proceed to batched computation
    if 'jacobian_single' in locals():
        # Use vmap to batch the Jacobian computation
        jacobian_func_batch = jax.vmap(jax.jacfwd(model_apply), in_axes=0, out_axes=0)
        jacobians = jacobian_func_batch(dummy_input)
        print("Jacobians shape:", jacobians.shape)
    else:
        print("Adjust the model or input processing before proceeding.")

    

    train(policy, train_data, epochs=400)

    print('fin')
