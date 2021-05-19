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


def reduce(nodes):
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    #accum = torch.cat((torch.mean(nodes.mailbox['m'], 1), torch.max(nodes.mailbox['m'], 1)[0]), dim=1)
    accum = torch.sum(alpha*nodes.mailbox['m'], dim=1)
    return {'hm': accum}

msg = fn.copy_src(src='h', out='m')

def message_func(edges):
    #return {'m': edges.src['h'] * edges.data['weights']}
    return {'m': edges.src['h'], 'e': edges.data['e']}
    #return {'m': edges.src['h']}

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        #self.linear = nn.Linear(3 * in_feats, out_feats)
        self.linear = nn.Linear(2*in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        # input dim = 6 because hm = 4 and h = 2
        h = self.linear(torch.cat([node.data['h'], node.data['hm']], dim=1))
        #h = self.linear(torch.cat([node.data['h']], dim=1))
        h = self.activation(h)
        return {'h': h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        self.attn_fc = nn.Linear(2*in_feats, 1, bias=False) #???

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def forward(self, g, feature):
        g.ndata['h'] = feature
        #g.update_all(msg, reduce) # Send messages along all edges of the specified type and update all the nodes of the corresponding destination type
        g.apply_edges(self.edge_attention)
        g.update_all(message_func, reduce)
        g.apply_nodes(func=self.apply_mod)
        g.ndata.pop('hm')
        return g.ndata.pop('h')

class ACNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(ACNet, self).__init__()

        self.policy = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 1)
        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])

    def forward(self, g):
        h = g.ndata['x']
        # we change the graph so we can pass the message along all vertices
        g = dgl.to_bidirected(g,True)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        mN = dgl.mean_nodes(g, 'h')
        PI = self.policy(g.ndata['h'])
        V = self.value(mN)
        g.ndata.pop('h')
        return PI, V


    '''
        def addfeature(self, g):
        x = g.ndata['hm']
        y = g.ndata['hm2']
        g.ndata['hm3'] = torch.cat((x[:, [2, 3]], y[:, [2, 1]]), 1)
        print(g.ndata['hm3'])
        
            x2w = nodes.mailbox['x2w']
    x1 = nodes.mailbox['x1out']
    print('x2w')
    print(x2w)
    print('x1')
    print(x1)
    
        trial = torch.cat((x1,x2w),2)
    print('trial')
    print(trial)
    print(trial.type())
    accum2 = torch.cat((torch.mean(trial, 1), torch.min(trial, 1)[0]), dim=1)
    print('accum')
    print(accum)
    print(accum.size())
    print('accum2')
    print(accum2)
    print('accum2 size')
    print(accum2.size())
    '''