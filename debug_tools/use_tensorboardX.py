from tensorboardX import SummaryWriter
import time
import numpy as np
import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import torch.nn as nn

from dgl.data import citation_graph as citegrh
import networkx as nx

from tensorboardX import SummaryWriter
writer = SummaryWriter('log/' + str(int(time.time())))

def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0  / len(labels)

def train(net):
    g, features, labels, train_mask, test_mask = load_cora_data()
    g.add_edges(g.nodes(), g.nodes())
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    dur = []
    t0 = 0

    for epoch in range(500):
        if epoch >= 3:
            t0 = time.time()

        net.train()
        logits = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(net, g, features, labels, test_mask)
        print("epoch {} | loss {} | acc {} | time {}".format(epoch, loss.item(), acc, np.mean(dur)))

        writer.add_scalar('Train/Loss', loss.item(), epoch)


# message function
gcn_msg = fn.copy_src(src='h', out='m')
# reduce function
gcn_reduce = fn.sum(msg='m', out='h')

class GCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.layer1 = GCNConv(1433, 16)
        self.layer2 = GCNConv(16, 7)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x=  self.layer2(g, x)
        return x

if __name__ == '__main__':
    model = GNN()
    train(model)

