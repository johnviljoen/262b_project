import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

def move_batch_to_device(batch, device="cpu"):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

class SimplifiedTrainer:
    def __init__(self, model, train_data, optimizer=None, train_metric="train_loss", clip=100.0, device="cpu"):
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=0.01)
        self.train_metric = train_metric
        self.clip = clip
        self.device = device

    def train(self, epochs=1):
        self.model.train()  # Set model to training mode
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in self.train_data:
                batch = move_batch_to_device(batch, self.device)
                
                self.optimizer.zero_grad()
                output = self.model(batch)  # Assuming batch contains 'input'
                loss = output[self.train_metric]  # Assuming model's output is a dict with self.train_metric
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f'Epoch {epoch}, Loss: {total_loss / len(self.train_data)}')

# Example usage
if __name__ == "__main__":
    # Assuming existence of a model, a data loader named train_data, and a loss function integrated within the model's output
    import torch
    from neuromancer.system import Node, System
    from neuromancer.modules import blocks
    from neuromancer.dataset import DictDataset
    from neuromancer.constraint import variable
    from neuromancer.loss import PenaltyLoss
    from neuromancer.problem import Problem
    from neuromancer.trainer import Trainer
    from neuromancer.plot import pltCL, pltPhase
    torch.manual_seed(1)

    # Double integrator parameters
    nx = 2
    nu = 1
    A = torch.tensor([[1.2, 1.0],
                      [0.0, 1.0]])
    B = torch.tensor([[1.0],
                      [0.5]])

    # closed loop system definition
    mlp = blocks.MLP(nx, nu, bias=True,
                     linear_map=torch.nn.Linear,
                     nonlin=torch.nn.ReLU,
                     hsizes=[20, 20, 20, 20])
    policy = Node(mlp, ['X'], ['U'])

    xnext = lambda x, u: x @ A.T + u @ B.T
    double_integrator = Node(xnext, ['X', 'U'], ['X'])
    cl_system = System([policy, double_integrator], init_func=lambda x: x)
    # cl_system.show()

    # Training dataset generation
    train_data = DictDataset({'X': 3.*torch.randn(3333, 1, nx)}, name='train')  # Split conditions into train and dev
    dev_data = DictDataset({'X': 3.*torch.randn(3333, 1, nx)}, name='dev')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                               collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                             collate_fn=dev_data.collate_fn, shuffle=False)

    # Define optimization problem
    u = variable('U')
    x = variable('X')
    action_loss = 0.0001 * (u == 0.)^2  # control penalty
    regulation_loss = 10. * (x == 0.)^2  # target position
    loss = PenaltyLoss([action_loss, regulation_loss], [])
    problem = Problem([cl_system], loss)
    # problem.show()

    trainer = SimplifiedTrainer(problem, train_loader)
    trainer.train(epochs=400)

    print('fin')
