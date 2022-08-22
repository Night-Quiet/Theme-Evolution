import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from function.func_ import pickle_load, cos_sim
import networkx as nx
import numpy as np


def train_(g_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matrix_ = nx.edges(g_)
    g_node = list(g_.nodes())
    node_num = {node_: ind for ind, node_ in enumerate(g_node)}
    edges_ = [[], []]
    for node1, node2 in matrix_:
        edges_[0].append(node_num[node1])
        edges_[1].append(node_num[node2])
    edges_ = np.array(edges_)
    matrix_ = torch.from_numpy(edges_).type(torch.LongTensor)

    model = Node2Vec(matrix_, embedding_dim=128, walk_length=20,
                     context_size=3, walks_per_node=100, num_negative_samples=3,
                     sparse=True, q=0.25).to(device)
    model = torch.nn.DataParallel(model)
    loader = model.module.loader(batch_size=128, shuffle=True, num_workers=8)
    optimizer = torch.optim.SparseAdam(model.module.parameters(), lr=0.01)


    def train():
        model.module.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.module.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)


    for epoch in range(100):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    torch.save(model, "./model.model")
    model.eval()
    z = model(torch.arange(len(g_node), device=device)).cpu().detach().numpy()

    return node_num, z


g_ = pickle_load("../model/all_node2vec.graph")
train_(g_)
model = torch.load("./model.model")
vectors = model(torch.arange(12338, device="cuda")).cpu().detach().numpy()

ha = 0.5
num = 0
for i in range(12338):
    for j in range(i+1, 12338):
        if cos_sim(vectors[i], vectors[j]) > ha:
            num += 1
            print(num)

print(num)


