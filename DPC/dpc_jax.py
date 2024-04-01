import jax
import jax.numpy as jnp
from flax import linen as nn  # Flax's neural network module
from flax.training import train_state  # Useful for managing training state
import optax  # Optimizer library for JAX
from typing import Any, Callable, Dict, Sequence
import numpy as np

# Helper function to move data to device in JAX (no need for this in JAX as operations are on the device by default)
def move_batch_to_device(batch):
    return {k: jnp.array(v) for k, v in batch.items()}

# Simulate the dataset as numpy arrays.
# In practice, you would load your actual dataset here.
np.random.seed(1)
train_x = 3.0 * np.random.randn(3333, 2)
train_y = train_x @ np.array([[1.2, 1.0], [0.0, 1.0]]).T + np.random.randn(3333, 2) * 0.1

# Convert the numpy arrays to JAX arrays.
train_x = jax.device_put(train_x)
train_y = jax.device_put(train_y)

# Simple batching function.
def get_batches(batch_size, *arrays):
    starts = np.arange(0, arrays[0].shape[0], batch_size)
    ends = starts + batch_size
    for start, end in zip(starts, ends):
        yield tuple(array[start:end] for array in arrays)

# Define the model.
class SimpleMLP(nn.Module):
    features: Sequence[int]
    output_dim: int  # Add an output dimension parameter

    @nn.compact
    def __call__(self, x):
        for feature in self.features:
            x = nn.Dense(feature)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)  # Output layer with specified output dimension
        return x


class SimplifiedTrainer:
    def __init__(self, model, train_data, optimizer=None, train_metric="train_loss", clip=100.0):
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer if optimizer else optax.adam(0.01)
        self.train_metric = train_metric
        self.clip = clip
        self.state = train_state.TrainState.create(apply_fn=model.apply, params=model.init(jax.random.PRNGKey(0), jnp.ones((1,))), tx=self.optimizer)

    def train(self, epochs=1):
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in self.train_data:
                batch = move_batch_to_device(batch)
                loss, grads = jax.value_and_grad(self.loss_function)(self.state.params, batch)
                grads = jax.tree_map(lambda g: jnp.clip(g, -self.clip, self.clip), grads)  # Gradient clipping
                self.state = self.state.apply_gradients(grads=grads)
                total_loss += loss
            
            print(f'Epoch {epoch}, Loss: {total_loss / len(self.train_data)}')

    def loss_function(self, params, batch):
        predictions = self.model.apply(params, batch['input'])
        loss = predictions[self.train_metric]  # Assuming model's output is a dict with self.train_metric
        return loss

# Example usage
if __name__ == "__main__":
    # the model with a sequence of layer sizes.
    model = SimpleMLP(features=[20, 20, 20, 20], output_dim=2)

    # Define the optimizer.
    optimizer = optax.adam(0.01)

    # Example training loop.
    batch_size = 333
    num_epochs = 400
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=model.init(jax.random.PRNGKey(0), train_x[:1]), tx=optimizer)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_batch, y_batch in get_batches(batch_size, train_x, train_y):
            # Define a loss function for your problem.
            # Here's an example using mean squared error.
            def mse_loss(params, x, y):
                pred_y = model.apply(params, x)
                return jnp.mean((pred_y - y) ** 2)

            loss, grads = jax.value_and_grad(mse_loss)(state.params, x_batch, y_batch)
            grads = jax.tree_map(lambda g: jnp.clip(g, -100.0, 100.0), grads)
            state = state.apply_gradients(grads=grads)
            total_loss += loss * x_batch.shape[0]

        total_loss /= train_x.shape[0]
        print(f'Epoch {epoch}, Loss: {total_loss}')

    print('fin')
