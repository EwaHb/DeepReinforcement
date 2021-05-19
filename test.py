# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:28:50 2019
@author: orrivlin
"""

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch import GATConv

# import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import time

# sys.setrecursionlimit(10**6)
'''
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'h': edges.src['h'], 'e': edges.data['weights']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)#1)normalize the attention scores -> the mailbox contains the attention scores
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['h'], dim=1)#Aggregate neighbor embeddings weighted by the attention scores (equation(4)).
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['h'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)#sends out 2 tensors - the transformed z embedding souce node and the unnormalized attention score on each edge
        #print(self.g.ndata.pop('h'))
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


'''


# msg = fn.copy_src(src='h', out='m')


class NodeApplyModule(nn.Module):
    '''
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(3 * in_feats, out_feats)
        self.activation = activation
        '''

    def __init__(self, g, in_dim, out_dim):
        super(NodeApplyModule, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(edges):
        # message UDF for equation (3) & (4)
        # return {'m': edges.src['h'] * edges.data['weights'], 'x2w': edges.src['x2'] * edges.data['weights'], 'x1out': edges.src['x1'] }
        # return {'m': edges.src['h'] * edges.data['weights']}#the message retuns the nodes`s source feature *weight - was before
        return {'m': edges.source['x'], 'e': edges.data['e']}

    def reduce(nodes):
        # accum = torch.cat((torch.mean(nodes.mailbox['m'], 1), torch.max(nodes.mailbox['m'], 1)[0]), dim=1)
        accum = F.softmax(nodes.mailbox['m'], dim=1)
        accum1 = torch.sum(accum * nodes.mailbox['x'], dim=1)
        return {'hm': accum1}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['x'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func,
                          self.reduce_func)  # sends out 2 tensors - the transformed z embedding souce node and the unnormalized attention score on each edge
        # print(self.g.ndata.pop('h'))
        return self.g.ndata.pop('h')


'''
    def forward(self, node):
        h = self.linear(torch.cat((node.data['x'], node.data['hm']), dim=1))
        h = self.activation(h)
        return {'h': h}
'''


class GCN(nn.Module):
    '''
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        '''

    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(GCN, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(NodeApplyModule(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


'''
    def forward(self, g, feature):
        z = self.fc(feature)
        self.g.ndata['x'] = z
        self.g.apply_rdges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
        #g.ndata['h'] = feature # we have a matrix of features and each node gets a line of features as its h value
        #g.update_all(msg, reduce) # Send messages along all edges of the specified type and update all the nodes of the corresponding destination type
        #g.update_all(message_func, reduce)
        #g.apply_nodes(func=self.apply_mod) # applies this function  (apply_mod) to all the nodes - makes sense
        #print(g.ndata['hm'])
        #g.ndata.pop('hm') # deleted hm as it has been already used/ applied when we applied a linear function to each node
        #return g.ndata.pop('h')
'''


class ACNet(nn.Module):
    '''
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(ACNet, self).__init__()

        self.policy = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 1)
        self.layers = nn.ModuleList([
            ACNet(in_dim, hidden_dim, F.relu),
            ACNet(hidden_dim, hidden_dim, F.relu),
            ACNet(hidden_dim, hidden_dim, F.relu)]) # we do not have an output dim here becasue we have a policy NN to return the final vector
'''

    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(ACNet, self).__init__()
        self.layer1 = GCN(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GCN(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, g):
        h = g.ndata['x']
        # h = g.edata['weights']

        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)

        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        # g.edata['weights']
        mN = dgl.mean_nodes(g,
                            'h')  # calculates the mean value of the features 'h' for one node e.g. if a node has these 'h' features: [1,2,3] it would be (1+2+3)/3 = 2
        # mN = dgl.mean_edges(g,'weights')
        PI = self.policy(g.ndata['h'])  # this is one dimension with 30 elements
        V = self.value(mN)
        g.ndata.pop('h')
        return PI, V

