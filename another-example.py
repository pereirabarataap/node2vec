import random

import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pandas as pd

random.seed(0)
np.random.seed(0)

G = nx.karate_club_graph()

def get_embedding():
    model = Node2Vec(G, dimensions=4, seed=0)
    embedding = model.fit()
    return pd.DataFrame(embedding.wv.vectors, index=embedding.wv.index2entity)

print('Embedding 1')
print(get_embedding().sort_index().head())

print('Embedding 2')
print(get_embedding().sort_index().head())