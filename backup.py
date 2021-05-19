import dgl
import torch
import networkx as nx
import numpy as np
import sys

class Graphs1:

    def __init__(self):
        self.src = (np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
             4,
             4,
             4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8,
             8,
             8,
             8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]))-1
        self.dst = (np.array(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16,
             17,
             18,
             19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15,
             16,
             17,
             18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14,
             15,
             16,
             17, 18, 19]))-1
        self.w = np.array(
            [2457, 638, 3176, 943, 1398, 902, 897, 292, 464, 443, 2409, 683, 3219, 904, 1347, 835, 830, 341, 413, 392, 2075,
             1515, 3420, 853, 1908, 1342, 1337, 1196, 1086, 1065, 2406, 839, 3375, 905, 1296, 809, 804, 497, 449, 428, 2301,
             806, 3302, 797, 1278, 728, 723, 449, 306, 285, 2205, 915, 3437, 723, 1130, 615, 610, 573, 374, 353, 3172, 1647,
             2252, 1658, 2452, 1898, 1893, 1353, 1484, 1463, 2480, 811, 2945, 942, 1616, 1062, 1057, 518, 648, 627, 2582,
             806, 3017, 1041, 1612, 1058, 1053, 513, 643, 622])

    @staticmethod
    def build_customed_graph_1February():

        # create source and destination nodes
        src = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4,
             4,
             4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
             8,
             8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9])
        src = src - 1
        dst = np.array(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 17,
             18,
             19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16,
             17,
             18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15,
             16,
             17, 18, 19])
        dst = dst - 1
        # create distances between source and destination nodes i.e. between trucks and customers
        w = np.array(
            [2457, 638, 3176, 943, 1398, 902, 897, 292, 464, 443, 2409, 683, 3219, 904, 1347, 835, 830, 341, 413, 392, 2075,
             1515, 3420, 853, 1908, 1342, 1337, 1196, 1086, 1065, 2406, 839, 3375, 905, 1296, 809, 804, 497, 449, 428, 2301,
             806, 3302, 797, 1278, 728, 723, 449, 306, 285, 2205, 915, 3437, 723, 1130, 615, 610, 573, 374, 353, 3172, 1647,
             2252, 1658, 2452, 1898, 1893, 1353, 1484, 1463, 2480, 811, 2945, 942, 1616, 1062, 1057, 518, 648, 627, 2582,
             806, 3017, 1041, 1612, 1058, 1053, 513, 643, 622])
        #w = w.astype(float)

        # normalize distances
        sum = np.sum(w)
        div = np.divide(w,sum)

        # create DGL graph
        g = dgl.graph((src, dst))
        weights = torch.from_numpy(div)
        weights = weights.reshape(g.number_of_edges(),1)
        g.edata['weights'] = torch.Tensor(np.array(weights))
        return g

    @staticmethod
    def get_allocation_plan():
        return


