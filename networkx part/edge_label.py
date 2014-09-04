#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx

G=nx.Graph()

a="hphob"
b="polarity"
c="alpha"
d="beta"
G.add_edge(a,b,weight=0.5)
G.add_edge(b,c,weight=0.5)
G.add_edge(c,d,weight=0.5)
G.add_edge(a,d,weight=0.5)
G.add_edge(a,c,weight=0.5)
G.add_edge(b,d,weight=0.5)

pos=nx.spring_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,node_size=7000, node_color="white")

# edges
nx.draw_networkx_edges(G,pos,
        width=6,alpha=0.5,edge_color='black')


# labels
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

nx.draw_networkx_edge_labels(G,pos, 
    {
        (a,b):"x", (b,c):"y", (c,d):"w", (a,d):"z", (a,c):"v", (b,d):"r"
    },
    label_pos=0.3
)

plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display