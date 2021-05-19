# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:00:48 2021

@author: PTC
"""
from networkx.algorithms import bipartite
import networkx as nx
import random
import matplotlib.pyplot as plt


random.seed(0)

def biased_random_connected_bipartite(n, m, k):
  G = nx.Graph()

  # These will be the two components of the bipartite graph
  N = set(range(n))
  M = set(range(n, n+m))
  G.add_nodes_from(N)
  G.add_nodes_from(M)
  #B.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])

  # Create a first random edge
  u = random.choice(tuple(N))
  v = random.choice(tuple(M))
  #res = random.sample(range(1, 50), 7)
  w = random.randint(0,50)
  print(w)
  G.add_edge(u, v,weight=w)
  #print(G.edges)

  isolated_N = set(N-{u})
  isolated_M = set(M-{v})
  while isolated_N and isolated_M:
    # Pick any isolated node:
    isolated_nodes = isolated_N|isolated_M
    u = random.choice(tuple(isolated_nodes))

    # And connected it to the existing connected graph:
    if u in isolated_N:
      v = random.choice(tuple(M-isolated_M))
      w = random.randint(0,50)
      #G.add_edge(u, v,)
      G.add_edge(u, v,weight=w)
      isolated_N.remove(u)
    else:
      v = random.choice(tuple(N-isolated_N))
      w = random.randint(0,50)
      G.add_edge(u, v,weight=w)
      isolated_M.remove(u)

  # Add missing edges
  for i in range(k-len(G.edges())):
    u = random.choice(tuple(N))
    v = random.choice(tuple(M))
    w = random.randint(0,50)
    G.add_edge(u, v,weight=w)
    G.add_edge(u, v)

  return G



G = biased_random_connected_bipartite(8, 8, 40)


pos = nx.spring_layout(G, scale=50)
nx.draw(G, pos,with_labels=True, font_weight='bold')
#edge_labels = nx.get_edge_attributes(G,'weight')
grafo_labels = nx.get_edge_attributes(G,'weight')
color_map = []
for node in G:
    if node < 8:
        color_map.append('blue')
    else:
        color_map.append('green')
nx.draw_networkx_nodes(G, pos, node_color=color_map)
nx.draw_networkx_edge_labels(G, pos, edge_labels = grafo_labels)

plt.title("Random Graph Generation Example")
plt.show()