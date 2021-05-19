import dgl
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys

class Graphs:

    @staticmethod
    def build_graph_of_coordinates():

        # create source and destination nodes
        src = np.array(
            [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
        src = src - 1
        dst = np.array(
            [5,6,7,8,9,5,6,7,8,9,5,6,7,8,9,5,6,7,8,9])
        dst = dst - 1
        # create distances between source and destination nodes i.e. between trucks and customers
        w = np.array(
            [3.16,3.6,5.1,5.1,7.1,7.1,2.23,3.16,7.61,3.16,7.1,8.1,3.16,5.1,5.1,4.1,5.1,2.24,3.6,5])



        # normalize distances
        sum = np.sum(w)
        div = np.divide(w,sum)

        # create DGL graph
        g = dgl.graph((src, dst))
        weights = torch.from_numpy(div)
        weights = weights.reshape(g.number_of_edges(),1)
        weights = weights*(-1)
        print('weights')
        print(weights)
        g.edata['weights'] = torch.Tensor(np.array(weights))
        return g

    @staticmethod
    def getNdata():
        return torch.Tensor(np.array([[(0.6),(0.2)],[(0.8),(0.6)],[(0.2),(0.8)],[(0.4),(0.5)],[(0.3),(0.1)],[(0.9),(0.4)],[(0.5),(0.7)],[(0.1),(0.3)],[(0.7),(0.9)]]))

    @staticmethod
    def get_cost(plan):
        w = np.array(
            [3.16,3.6,5.1,5.1,7.1,7.1,2.23,3.16,7.61,3.16,7.1,8.1,3.16,5.1,5.1,4.1,5.1,2.24,3.6,5])
        plan = plan.tolist()
        accum = 0
        for i in plan:
            accum+=w[i]
        return accum

    @staticmethod
    def get_allocation_plan(plan):
        src = np.array(
            [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
        src = src - 1
        dst = np.array(
            [5,6,7,8,9,5,6,7,8,9,5,6,7,8,9,5,6,7,8,9])
        dst = dst - 1

        a_zip = zip(src, dst)

        zipped_list = list(a_zip)

        accum = []
        for i in plan:
            accum.append(zipped_list[i])

        return accum

'''
g =  Graphs.build_graph_of_coordinates()
print(g.number_of_edges)
print(g.edata)
bg1 = dgl.to_bidirected(g,True)
print(bg1.number_of_edges)
print(bg1.edata)

nx.draw(g.to_networkx())
plt.show()
'''