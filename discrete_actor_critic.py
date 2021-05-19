
import torch
import numpy as np
import dgl
import torch.nn.functional as F
from copy import deepcopy as dc
from Models import ACNet
from Utils import stack_indices, stack_list_indices, max_graph_array
from log_utils import mean_val, logger
#from DSVGraphs import Graphs
from coordinatesGraph import Graphs


class DiscreteActorCritic:
    def __init__(self, problem, cuda_flag):
        self.problem = problem
        ndim = self.problem.get_graph_dims()
        if cuda_flag:
            self.model = ACNet(ndim, 264, 1).cuda()
        else:
            #128
            #264
            self.model = ACNet(ndim, 264, 1)
        self.gamma = 0.98
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.batch_size = 32
        self.num_episodes = 1
        self.cuda = cuda_flag
        self.log = logger()
        self.log.add_log('tot_return')
        self.log.add_log('TD_error')
        self.log.add_log('entropy')

    def run_episode(self):

        #print('METHOD: run_episode Outer Loop')
        sum_r = 0
        state, done = self.problem.reset()
        truckLoc = self.problem.get_random_truck(state)

        action = None

        [idx1, idx2] = self.problem.get_ilegal_actions(state, action)
        t = 0

        while done == False:

            G = dc(state.g)
            if self.cuda:
                G.ndata['x'] = G.ndata['x'].cuda()
            [pi, val] = self.model(G)
            pi = pi.squeeze()
            pi[idx1] = -float('Inf')
            pi = F.softmax(pi, dim=0)
            print('weights')
            print(np.sum(self.model.policy.weight.data.numpy()))

            if torch.all(pi.isnan()):
                plan = ((state.edges == 1.).nonzero()).squeeze()
                print('allocation cost')
                print(Graphs.get_cost(plan))
                print('allocation plan')
                print(Graphs.get_allocation_plan(plan))
                print(self.log)
                break
            dist = torch.distributions.categorical.Categorical(pi)
            oldAction = action
            #action = dist.sample()
            action = torch.argmax(pi)

            new_state, reward, done = self.problem.step(dc(state), action, oldAction)
            [idx1, idx2] = self.problem.get_ilegal_actions(new_state, action)
            state = dc(new_state)
            sum_r += reward
            if (t == 0):
                PI = pi[action].unsqueeze(0)
                R = reward.unsqueeze(0)
                V = val.unsqueeze(0)
                t += 1
            else:
                PI = torch.cat([PI, pi[action].unsqueeze(0)], dim=0)
                R = torch.cat([R, reward.unsqueeze(0)], dim=0)
                V = torch.cat([V, val.unsqueeze(0)], dim=0)

        self.log.add_item('tot_return', sum_r.item())
        tot_return = R.sum().item()
        #for i in range(R.shape[0] - 1):
        #  R[-2 - i] = R[-2 - i] + self.gamma * R[-1 - i]

        return PI, R, V, tot_return

    def update_model(self, PI, R, V):
        #print('update model')
        #print('PI')
        #print(PI)
        B = torch.tensor([0.0, -0.0329, 0.0, -0.0232, 0.0, -0.0329, 0.0, -0.0375])
        self.optimizer.zero_grad()
        if self.cuda:
            R = R.cuda()

        #A = R.squeeze() - V.squeeze().detach()
        A = R.squeeze() - B.squeeze().detach()
        # instead of mean we need to do sum
        L = -(torch.log(PI) * A).sum()
        #just to prevent the error
        L_value = F.smooth_l1_loss(V.squeeze(), R.squeeze())
        L_entropy = -(PI * PI.log()).mean()
        '''
        L_policy = -(torch.log(PI) * A).mean()
        print('L_policy')
        print(L_policy)
        L_value = F.smooth_l1_loss(V.squeeze(), R.squeeze())
        L_entropy = -(PI * PI.log()).mean()
        L = L_policy + L_value - 0.1 * L_entropy
        '''
        L.backward()
        self.optimizer.step()
        self.log.add_item('TD_error', L_value.detach().item())
        self.log.add_item('entropy', L_entropy.cpu().detach().item())

    def train(self):
        mean_return = 0
        for i in range(self.num_episodes):
            [pi, r, v, tot_return] = self.run_episode()
            mean_return = mean_return + tot_return
            if (i == 0):
                PI = pi
                R = r
                V = v
            else:
                PI = torch.cat([PI, pi], dim=0)
                R = torch.cat([R, r], dim=0)
                V = torch.cat([V, v], dim=0)

        mean_return = mean_return / self.num_episodes
        self.update_model(PI, R, V)
        return self.log
'''

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:20:39 2018
@author: orrivlin
"""
import torch
import numpy as np
import dgl
import torch.nn.functional as F
from copy import deepcopy as dc
from Models import ACNet
from Utils import stack_indices, stack_list_indices, max_graph_array
from log_utils import mean_val, logger
# from DSVGraphs import Graphs
from coordinatesGraph import Graphs
import math


class DiscreteActorCritic:
    def __init__(self, problem, cuda_flag):
        self.problem = problem
        ndim = self.problem.get_graph_dims()
        if cuda_flag:
            self.model = ACNet(ndim, 128, 1).cuda()
        else:
            # 128
            # 264
            self.model = ACNet(ndim, 128, 1)
        self.gamma = 0.98
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.batch_size = 32
        self.num_episodes = 1
        self.cuda = cuda_flag
        self.log = logger()
        self.log.add_log('tot_return')
        self.log.add_log('TD_error')
        self.log.add_log('entropy')

    def run_episode(self):

        # print('METHOD: run_episode Outer Loop')
        sum_r = 0
        state, done = self.problem.reset()
        truckLoc = self.problem.get_random_truck(state)

        action = None

        [idx1, idx2] = self.problem.get_ilegal_actions(state, action)
        print('idx1,idx2')
        print(idx1, idx2)
        t = 0

        while done == False:

            G = dc(state.g)
            if self.cuda:
                G.ndata['x'] = G.ndata['x'].cuda()
            # [pi, val] = self.model(G)
            pi = self.model(G)

            print('pi')
            print(pi)
            pi = pi.squeeze()
            pi[idx1] = -float('Inf')
            print('pi after -inf')
            print(pi)
            pi = F.softmax(pi, dim=0)

            if torch.all(pi.isnan()):
                plan = ((state.edges == 1.).nonzero()).squeeze()
                print('allocation cost')
                print(Graphs.get_cost(plan))
                print('allocation plan')
                print(Graphs.get_allocation_plan(plan))
                print(self.log)
                break
            dist = torch.distributions.categorical.Categorical(pi)
            oldAction = action
            # action = dist.sample()
            action = torch.argmax(pi)
            print('action again')
            print(action)

            new_state, reward, done = self.problem.step(dc(state), action, oldAction)
            [idx1, idx2] = self.problem.get_ilegal_actions(new_state, action)
            state = dc(new_state)
            sum_r += reward
            if (t == 0):
                PI = pi[action].unsqueeze(0)
                R = reward.unsqueeze(0)
                # V = val.unsqueeze(0)
                [idz1, idz2] = (action, R)
                setActions = []
                setActions.append([idz1, idz2])
                print('(action')
                print(action)
                print('R')
                print(R)
                print('idz1,idz2')
                print(idz1, idz2)
                print('setActions')
                print(setActions)
                t += 1
            else:
                PI = torch.cat([PI, pi[action].unsqueeze(0)], dim=0)
                R = torch.cat([R, reward.unsqueeze(0)], dim=0)
                # V = torch.cat([V, val.unsqueeze(0)], dim=0)
                [idz1, idz2] = (action, R)
                setActions = []
                setActions.append([idz1, idz2])
                print('(action')
                print(action)
                print('R')
                print(R)
                print('idz1,idz2')
                print(idz1, idz2)
                print('setActions')
                print(setActions)

        self.log.add_item('tot_return', sum_r.item())
        tot_return = R.sum().item()

        for i in range(R.shape[0] - 1):
            R[-2 - i] = R[-2 - i] + self.gamma * R[-1 - i]

        # return PI, R, V, tot_return
        return PI, R, tot_return, setActions


    # loss += criterion(decoder_output, target_tensor[di])
    # def update_model(self, PI, R, V):
    def update_model(self, PI, R):
        b = 12.15
        # print('update model')
        print('PI')
        print(PI)
        self.optimizer.zero_grad()
        if self.cuda:
            R = R.cuda()
        print('R')
        print(R)
        # A = R.squeeze() - V.squeeze().detach()
        # A = R.squeeze().detach() # this is wrong
        A = torch.argmax(PI)
        print('A')
        print(A)
        L_policy = -(torch.log(PI) * A).mean()
        # print('L_policy')
        # print(L_policy)
        # L_value = F.smooth_l1_loss(V.squeeze(), R.squeeze())
        # print('L_value.detach().item')
        # print(L_value.detach().item())
        L_entropy = -(PI * PI.log()).mean()
        # L = L_policy + L_value - 0.1 * L_entropy
        # L = torch.Tensor((loss* math.log(PI(A))) * L_entropy)

        [idy1, idy2, idy3, idy4] = self.run_episode()
        print('idy1')
        print(idy1)
        print('idy2')
        print(idy2)
        print('idy3')
        print(idy3)
        print('idy4')
        print(idy4)

        actionsANDreward = idy4
        print('actionsANDreward')
        print(actionsANDreward)
        L = 0
        for index, tuple in enumerate(actionsANDreward):
            element_oneAction = tuple[0]
            print('element_oneAction')
            print(element_oneAction)
            # print(PI(element_oneAction))
            element_twoReward = tuple[1]
            print('element_twoReward')
            print(element_twoReward)
            # print('PI(element_oneAction)')
            # print(PI(element_oneAction))
            L = L + ((element_twoReward - b) * PI)
            print('L')
            print(L)
        L = torch.sum(L)
        print('L again')
        print(L)

        L.backward()
        self.optimizer.step()
        # self.log.add_item('TD_error', L_value.detach().item())
        self.log.add_item('entropy', L_entropy.cpu().detach().item())

    def train(self):
        b = 12.15
        mean_return = 0
        for i in range(self.num_episodes):
            # [pi, r, v, tot_return] = self.run_episode()
            [pi, r, tot_return, setActions] = self.run_episode()
            mean_return = mean_return + tot_return
            if (i == 0):
                PI = pi
                R = r
            # V = v
            else:
                PI = torch.cat([PI, pi], dim=0)
                R = torch.cat([R, r], dim=0)
                # V = torch.cat([V, v], dim=0)
        #
        mean_return = mean_return / self.num_episodes
        # self.update_model(PI, R, V)
        self.update_model(PI, R)
        return self.log
'''

