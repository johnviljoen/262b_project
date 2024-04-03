"""
This file is a clean torch implementation of the DPC algorithm
"""

import torch
import torch.nn as nn
torch.manual_seed(1)

def dynamics(state, action):
    A = torch.tensor([[1.2, 1.0],
                    [0.0, 1.0]])
    B = torch.tensor([[1.0],
                    [0.5]])
    return state @ A.T + action @ B.T

def cost(state, action):
    # batch cost, dotted across final dimension and summed across first
    state_loss = torch.einsum('ijk,ilk->jl', state, state)
    action_loss = torch.einsum('ijk,ilk->jl', action, action)
    return (0.0001 * action_loss + 10.0 * state_loss)/state.shape[0]

class MLP(nn.Module):
    def __init__(self, nx, nu, bias=True, linear_map=nn.Linear, nonlin=nn.ReLU, hsizes=[20, 20, 20, 20]):
        super(MLP, self).__init__()
        layers = []
        in_dim = nx
        for hsize in hsizes:
            layers.append(linear_map(in_dim, hsize, bias=bias))
            layers.append(nonlin())
            in_dim = hsize
        layers.append(linear_map(in_dim, nu, bias=bias))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
class Trainer:
    def __init__(self, policy, train_data, optimizer=None, clip=100.0):
        self.policy = policy # assumes .train() has been called
        self.train_data = train_data
        self.optimizer = optimizer if optimizer else torch.optim.Adam(policy.parameters(), lr=0.01)
        self.clip = clip

    def train(self, epochs=1):
        for epoch in range(epochs):
            total_loss = 0.0
            for state in self.train_data:
                self.optimizer.zero_grad()
                action = self.policy(state)
                next_state = dynamics(state, action)
                loss = cost(next_state, action)
                loss.backward()

                # gradient clipping
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip)

                self.optimizer.step()
                total_loss += loss.detach()
            
            print(f'Epoch {epoch}, Loss: {total_loss / len(self.train_data)}')

if __name__ == "__main__":
    nx = 2
    nu = 1
    train_data = 3.*torch.randn(1, 3333, 1, nx)
    policy = MLP(nx, nu).train()
    trainer = Trainer(policy, train_data)
    trainer.train(epochs=400)

    print('fin')
