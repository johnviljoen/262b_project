"""this one will be the first attempt to remove the whole neuromancer dependency successfully
result: it actually works! great success!"""

import torch
import torch.nn as nn
torch.manual_seed(1)
from neuromancer.dataset import DictDataset

def move_batch_to_device(batch, device="cpu"):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

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

class TrainerExtracted:
    def __init__(self, model, train_data, optimizer=None, train_metric=None, clip=100.0, device="cpu"):
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=0.01)
        self.train_metric = cost # cost function
        self.clip = clip
        self.device = device

    def train(self, epochs=1):
        self.model.train()  # Set model to training mode
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in self.train_data:
                batch = move_batch_to_device(batch, self.device)
                
                self.optimizer.zero_grad()
                u = self.model(batch['X'])  # Assuming batch contains 'input'
                x_next_predicted = dynamics(batch['X'], u)  # Predict next state using dynamics

                loss = self.train_metric(x_next_predicted, u)  # Assuming model's output is a dict with self.train_metric
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f'Epoch {epoch}, Loss: {total_loss / len(self.train_data)}')


# Double integrator parameters
nx = 2
nu = 1
A = torch.tensor([[1.2, 1.0],
                    [0.0, 1.0]])
B = torch.tensor([[1.0],
                    [0.5]])

# Training dataset generation
train_data = DictDataset({'X': 3.*torch.randn(3333, 1, nx)}, name='train')  # Split conditions into train and dev
dev_data = DictDataset({'X': 3.*torch.randn(3333, 1, nx)}, name='dev')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                            collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                            collate_fn=dev_data.collate_fn, shuffle=False)

def cost(x, u):
    # batch cost, dotted across final dimension and summed across first
    x_loss = torch.einsum('ijk,ilk->jl', x, x)
    u_loss = torch.einsum('ijk,ilk->jl', u, u)
    return (0.0001 * u_loss + 10.0 * x_loss)/x.shape[0]

policy = MLP(nx, nu)
dynamics = lambda x, u: x @ A.T + u @ B.T

policy.train()


# Example usage
if __name__ == "__main__":
    # Assuming existence of a model, a data loader named train_data, and a loss function integrated within the model's output
    import torch


    trainer = TrainerExtracted(policy, train_loader)
    trainer.train(epochs=400)

    print('fin')
