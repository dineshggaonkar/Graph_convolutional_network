import torch
import config
import pandas as pd

from model import GCN
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet

data = MoleculeNet(root=".", name="ESOL")

test_loader = DataLoader(data[int(config.data_size * 0.9):],
                         batch_size=1, shuffle=True)
test_batch = next(iter(test_loader))

model = GCN()
model.load_state_dict(torch.load("./Gcn_model/graph_classif_model.pb"))
model.eval()

'''
#batch prediction (change batch size)

with torch.no_grad():
    test_batch.to(device)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()
'''

#solubility prediction for single molecule (batch size = 1)

with torch.no_grad():
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
    y_real = test_batch.y.tolist()
    y_pred = pred.tolist()
    print(f"ground truth = {y_real}  ---  predicted value = {y_pred}")
