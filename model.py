import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.cls = GCNConv(self.nhid, self.num_classes)

    def forward(self, data, edge_index):
        x = self.feat_bottleneck(data, edge_index)
        x = self.feat_classifier(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def feat_bottleneck(self, data, edge_index):
        x = F.relu(self.conv1(data.x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x
    
    def feat_classifier(self, x, edge_index):
        x = self.cls(x, edge_index)
        
        return x