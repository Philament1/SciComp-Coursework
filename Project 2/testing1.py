import networkx as nx
import matplotlib.pyplot as plt
from project2 import searchGPT, searchPKR, searchPKR2
import numpy as np
from numpy import random

#random.seed(1)


def weighted_ba_graph(n, m, w_max):
    G = nx.barabasi_albert_graph(n, m)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = random.randint(1, w_max)
    return G

def custom_graph():
    
    G = nx.Graph()
    nodes = [0, 1, 2, 3, 4, 5]
    G.add_nodes_from(nodes)

    # Add edges with weights
    edges_with_weights = [(0, 1, 1), (1, 2, 2), (2, 3, 1), (0, 3, 3), (0, 4, 1), (0, 5, 2), (5, 4, 2), (4, 3, 2)]
    G.add_weighted_edges_from(edges_with_weights)

    return G

def disconnected_graph():

    G = nx.Graph()
    nodes = [0, 1, 2, 3, 4, 5]

    G.add_nodes_from(nodes)

    # Add edges with weights
    edges_with_weights = [(0, 1, 1), (1, 2, 5), (0, 2, 3), (3, 4, 1), (3, 5, 2), (4, 5, 3)]
    G.add_weighted_edges_from(edges_with_weights)

    return G

#G = weighted_ba_graph(10, 2, 16)

G = disconnected_graph()
G_ = G.copy()

search1 = searchGPT(G_, 0, 5)

search2 = searchPKR2(G_, 0, 5)


print(search1)
print(search2)

plt.figure()
pos = nx.spring_layout(G_)
nx.draw(G_, pos, with_labels=True, font_weight='bold')
edge_labels = nx.get_edge_attributes(G_, 'weight')
nx.draw_networkx_edge_labels(G_, pos, edge_labels=edge_labels)

plt.show()


