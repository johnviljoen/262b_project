import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt

# Define the model
class SineNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# Implement the Adam optimizer
class AdamOptimizer:
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        self.v = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        self.m = jax.tree_map(lambda m, g: self.beta1 * m + (1 - self.beta1) * g, self.m, grads)
        self.v = jax.tree_map(lambda v, g: self.beta2 * v + (1 - self.beta2) * (g ** 2), self.v, grads)
        
        m_hat = jax.tree_map(lambda m: m / (1 - self.beta1 ** self.t), self.m)
        v_hat = jax.tree_map(lambda v: v / (1 - self.beta2 ** self.t), self.v)
        
        params = jax.tree_map(
            lambda p, m, v: p - self.learning_rate * m / (jnp.sqrt(v) + self.epsilon), 
            params, m_hat, v_hat
        )
        return params

# Define the loss function
@jax.jit
def mse_loss(params, inputs, targets):
    predictions = SineNet().apply({'params': params}, inputs)
    return jnp.mean((targets - predictions) ** 2)

# Generate synthetic data
rng = jax.random.PRNGKey(0)
inputs = jnp.linspace(-jnp.pi, jnp.pi, 100).reshape(-1, 1)
targets = jnp.sin(inputs)

# Initialize model parameters
model = SineNet()
params = model.init(rng, inputs)['params']

# Initialize the Adam optimizer
adam_optimizer = AdamOptimizer(params, learning_rate=0.001)

# Define the number of epochs
num_epochs = 1000

# Training loop
for epoch in range(num_epochs):
    # Compute the gradients
    grads = jax.grad(mse_loss)(params, inputs, targets)
    # Update the parameters
    params = adam_optimizer.update(params, grads)
    # Compute loss (for printing purposes)
    loss = mse_loss(params, inputs, targets)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Plot the results
preds = SineNet().apply({'params': params}, inputs)
plt.plot(inputs.flatten(), targets.flatten(), label='Original Sine')
plt.plot(inputs.flatten(), preds.flatten(), label='NN Approximation')
plt.legend()
plt.show()
