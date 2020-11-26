import dgl
import dgl.function as fn
import torch
import torch.nn.function as F
from dgl import DGLGraph

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

model = GNN()
