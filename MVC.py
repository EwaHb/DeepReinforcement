# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:01:56 2019
@author: Or
"""

import dgl
import torch
import networkx as nx
import numpy as np
import sys

#from DSVGraphs import Graphs

from coordinatesGraph import Graphs


class State:
    def __init__(self, g, visited, edges):
        self.N = g.number_of_nodes()
        self.g = g
        self.visited = visited
        self.edges = edges


def init_state(N, P):
    g = Graphs.build_graph_of_coordinates()
    norm_card = torch.Tensor(np.array(g.in_degrees() + g.out_degrees()) / g.number_of_nodes()).unsqueeze(-1)
    #g.ndata['x'] = torch.cat((torch.zeros((g.number_of_nodes(), 1)), torch.ones((g.number_of_nodes(), 1))), dim=1)
    g.ndata['x'] = Graphs.getNdata()
    visited = torch.zeros((1, g.number_of_nodes())).squeeze()
    edges = torch.zeros((1, g.number_of_edges())).squeeze()
    return g, visited, edges


class MVC:
    def __init__(self, N, P):
        self.N = N
        self.P = P
        [g, visited, edges] = init_state(self.N, self.P)
        self.init_state = State(g, visited, edges)

    def get_graph_dims(self):
        return 2

    def reset(self):
        [g, visited, edges] = init_state(self.N, self.P)
        state = State(g, visited, edges)
        done = False
        return state, done

    def reset_fixed(self):
        done = False
        return self.init_state, done


    def get_ilegal_actionsORIGINAL(self,state,action):

        idx1 = (state.visited == 1.).nonzero()
        idx2 = (state.visited == 0.).nonzero()

        return idx1, idx2

    def get_random_truck(self,state):

        trucks = (state.g.in_degrees() == 0.).nonzero().squeeze()

        trucksS = torch.tensor(trucks.size())
        trucksI = trucksS.item()
        truckLoc = (torch.randint(0, trucksI, (1,))).squeeze()

        return truckLoc

    def get_ilegal_actions(self,state,action):

        idx1 = (state.visited == 1.).nonzero()
        #idx2 = (state.visited == 0.).nonzero()
        x = torch.zeros((1, 30)).squeeze()
        x = (x == 0.).nonzero()
        x = x.flatten().tolist()

        nodesIDs = torch.zeros((1, state.g.number_of_nodes())).squeeze()
        allActions = (nodesIDs == 0.).nonzero()
        allActions = allActions.flatten().tolist()

        legalActions = self.get_legal_actions(state, action)
        legalActions = legalActions.tolist()
        illegalActions = list(set(allActions) - set(legalActions))


        legalActionsT = torch.tensor(legalActions)
        legalActionsS = torch.tensor(legalActionsT.size())
        legalActionsI = legalActionsS.item()
        legalActions = (torch.reshape(legalActionsT, (legalActionsI ,1))).long()


        illegalActionsT = torch.Tensor(illegalActions)
        illegalActionsS = torch.tensor(illegalActionsT.size())
        illegalActionsI = illegalActionsS.item()
        illegalActions = torch.reshape(illegalActionsT, (illegalActionsI ,1)).long()

        return illegalActions, legalActions

    def get_legal_actions(self, state, action):
        #print('METHOD: get_legal_actions')

        visited_nodes = (state.visited == 1.).nonzero().squeeze()
        trucks = (state.g.in_degrees() == 0.).nonzero().squeeze()

        not_visited_truck_locations = self.intersection(visited_nodes, trucks)

        #now we need to get a list of nodes that we can travel from the action node
        x, y = state.g.edges()
        if action != None:
            out_nodes = list(zip(x.tolist(), y.tolist()))
            out_nodes = self.get_neighbours(out_nodes, action)
        else:
            out_nodes = torch.Tensor([])
        out_nodes = torch.reshape(out_nodes, (-1,))

        if action==None:
            #print('action==None')
            legal = (state.g.in_degrees() == 0.).nonzero().squeeze()
        #elif out_nodes.size()==0:
        elif len(out_nodes) == 0:
            #print('out_nodes.size()==0')
            legal = not_visited_truck_locations
        else:
            #print('else')
            legal = self.subtract(out_nodes,visited_nodes)

        return legal

    def intersection(self,t1,t2):
        compareview = t1.repeat(t2.shape[0],1).T
        intersection = t2[(compareview != t2).T.prod(1)==1]
        return intersection

    # subtrack t1 from t2
    def subtract(self,t1, t2):
        compareview = t2.repeat(t1.shape[0], 1).T
        return (t1[(compareview != t1).T.prod(1) == 1])

    def get_neighbours(self,list_of_tuples, action):
        out_edges = []

        for index, tuple in enumerate(list_of_tuples):
            element_one = tuple[0]
            element_two = tuple[1]
            if element_one == action:
                out_edges.append(element_two)
            else:
                pass
        out_edges = torch.Tensor(out_edges)
        out_edges = (out_edges.unsqueeze(-1)).to(torch.long)
        return out_edges

    def step(self, state, action, oldAction):
        reward = torch.Tensor(np.array([0])).squeeze()
        if oldAction==None:
            pass
        else:
            try:
                # I am assigning 1 to the visited edges
                edgeID = state.g.edge_ids(oldAction.item(), action.item())
                state.edges[edgeID] = 1
                #state.edges.append(edgeID)
                reward = ((state.g.edata['weights'][edgeID])).squeeze()
            except:
                print("No such edge connection error:", sys.exc_info()[0])
                pass

        done = False

        state.g.ndata['x'][action, 0] = 1.0 - state.g.ndata['x'][action, 0]

        state.visited[action.item()] = 1.0


        edge_visited = \
        torch.cat((state.visited[state.g.edges()[0]].unsqueeze(-1), state.visited[state.g.edges()[1]].unsqueeze(-1)),
                  dim=1).max(dim=1)[0]
        nodes_visited = (state.visited).tolist()

        if self.get_legal_actions(state,action).size()==0:
            done = True
        print('reward in the step method')
        print(reward)
        return state, reward, done
