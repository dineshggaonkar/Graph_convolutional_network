from torch_geometric.datasets import MoleculeNet


embedding_size = 64
NUM_GRAPHS_PER_BATCH = 64
n_epochs_stop = 50

data = MoleculeNet(root=".", name="ESOL")

print("Dataset type: ", type(data))
print("Dataset features: ", data.num_features)
print("Dataset target: ", data.num_classes)
print("Dataset length: ", data.len)
print("Dataset sample: ", data[0])
print("Sample  nodes: ", data[0].num_nodes)
print("Sample  edges: ", data[0].num_edges)

num_features = data.num_features
data_size = len(data)
