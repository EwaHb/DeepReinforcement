# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:30:53 2019
@author: orrivlin
"""

from discrete_actor_critic import DiscreteActorCritic
from MVC import MVC
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch

n = 19  # number of nodes
p = 0.15  # edge probability
env = MVC(n, p)
cuda_flag = False
alg = DiscreteActorCritic(env, cuda_flag)

num_episodes = 1
lista = []
for i in range(num_episodes):
    T1 = time.time()
    log = alg.train()
    T2 = time.time()
    lista.append(log.get_current('tot_return'))
    print('---------------------------------------------------')
    print('Epoch: {}. R: {}. TD error: {}. H: {}. T: {}'.format(i, np.round(log.get_current('tot_return'), 2),
                                                                np.round(log.get_current('TD_error'), 3),
                                                                np.round(log.get_current('entropy'), 3),
                                                                np.round(T2 - T1, 3)))

print(lista)
plt.plot(lista)
plt.show()
'''
Y = np.asarray(log.get_log('TD_error'))
print('td error')
print(Y)
plt.plot(Y)
plt.show()
'''



'''
Y = np.asarray(log.get_log('tot_return'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('episode return')

Y = np.asarray(log.get_log('TD_error'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('mean TD error')

Y = np.asarray(log.get_log('entropy'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('mean entropy')

PATH = 'mvc_net.pt'
torch.save(alg.model.state_dict(), PATH)
'''