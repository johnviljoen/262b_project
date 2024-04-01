import torch
import torch.nn as nn
torch.manual_seed(0)

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[20, 20, 20, 20]):
        super(PolicyNetwork, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def double_integrator_dynamics(x, u):
    A = torch.tensor([[1.2, 1.0], [0.0, 1.0]], dtype=torch.float32)
    B = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
    x_next = x @ A.T + u @ B.T
    return x_next

def rollout_and_optimize(model, optimizer, initial_state, nsteps, clip=100.0):
    # Store trajectory without in-place operations
    states = [initial_state.unsqueeze(0)]  # Wrap initial state in a list
    actions = []

    # Rollout trajectory
    for t in range(nsteps):
        current_state = states[-1]  # Get the last state
        action = model(current_state)
        next_state = double_integrator_dynamics(current_state, action)

        # Store state and action using list append to avoid in-place operations
        states.append(next_state)
        actions.append(action)

    # Convert lists to tensors
    states_tensor = torch.cat(states, dim=0)
    actions_tensor = torch.cat(actions, dim=0)

    # Compute loss over the trajectory
    action_loss = 0.0001 * torch.sum(actions_tensor ** 2)
    regulation_loss = 10. * torch.sum(states_tensor ** 2)
    loss = action_loss + regulation_loss

    # Backpropagation through time (BPTT)
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Perform optimization step
    optimizer.step()

    return loss.item()


def train(model, optimizer, epochs=800, nsteps=2, nx=2, nu=1):
    for epoch in range(epochs):
        # Generate a random initial state
        initial_state = 3. * torch.randn(nx)
        
        # Perform a rollout and optimize
        loss = rollout_and_optimize(model, optimizer, initial_state, nsteps)
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

# Initialize model, optimizer
policy_model = PolicyNetwork(2, 1)
policy_model.train()
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=0.001)

# Train
train(policy_model, optimizer)

print('fin')