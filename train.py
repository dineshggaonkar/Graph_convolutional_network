import torch
import numpy as np
import config

from torch_geometric.datasets import MoleculeNet
from model import GCN
from torch_geometric.data import DataLoader

# Load the ESOL dataset
data = MoleculeNet(root=".", name="ESOL")

model = GCN()
print(model)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)  

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loader = DataLoader(data[:int(config.data_size * 0.8)], 
                    batch_size=config.NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(data[int(config.data_size * 0.8):], 
                         batch_size=config.NUM_GRAPHS_PER_BATCH, shuffle=True)


def train():
    train_loss = 0
    # Enumerate over the data
    for batch in loader:
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        # Calculating the loss and gradients
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        # Update using the gradients
        optimizer.step()
        train_loss += loss.item()
    return train_loss/config.NUM_GRAPHS_PER_BATCH


print("Starting training...")

min_train_loss = np.Inf
train_losses = []

for epoch in range(2000):
    train_loss = 0
    loss = train()
    
    train_losses.append(loss)

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Train Loss {loss}")

    if round(train_losses[-1], 2) < min_train_loss:
        epochs_no_improve = 0
        min_train_loss = round(train_losses[-1], 2)
    else:
        epochs_no_improve += 1

    if epoch > 500 and epochs_no_improve == config.n_epochs_stop:
        print('Early stopping!')
        PATH = "./Gcn_model/graph_classif_model.pb"
        torch.save(model.state_dict(), PATH)
        print("model saved")
        
        break
    else:
        continue
