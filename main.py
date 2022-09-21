import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# import scipy as sp

data = pd.read_csv('fraudTrain.csv')
# print(data.head())
df1 = data[data["is_fraud"] == 0].sample(frac=0.20, random_state=42)
# print(df1.head())
df2 = data[data["is_fraud"] == 1]
# print(df2.head())
df = pd.concat([df1, df2])
# print(df.head())

# df.info()

print(df["is_fraud"].value_counts())

def build_graph_bipartite(df_input, graph_type=nx.Graph()):
    df = df_input.copy()
    mapping = {x:node_id for node_id, x in enumerate(set(df["cc_num"].values.tolist()
                                                         + df["merchant"].values.tolist()))}
    df["from"] = df["cc_num"].apply(lambda x: mapping[x])
    df["to"] = df["merchant"].apply(lambda x: mapping[x])
    df = df[['from', 'to', "amt", "is_fraud"]].groupby(['from', 'to']).agg({"is_fraud": "sum",
                                                                            "amt": "sum"}).reset_index()
    df["is_fraud"] = df["is_fraud"].apply(lambda x: 1 if x > 0 else 0)

    G = nx.from_edgelist(df[["from", "to"]].values, create_using=graph_type)

    nx.set_node_attributes(G, {x: 1 for x in df["from"].unique()}, "bipartite")
    nx.set_node_attributes(G, {x: 2 for x in df["to"].unique()}, "bipartite")

    nx.set_edge_attributes(G,
                           {(int(x["from"]), int(x["to"])): x["is_fraud"] for idx, x in
                            df[["from", "to", "is_fraud"]].iterrows()},
                           "label")
    nx.set_edge_attributes(G,
                           {(int(x["from"]), int(x["to"])): x["amt"] for idx, x in
                            df[["from", "to", "amt"]].iterrows()},
                           "weight")
    return G

def build_graph_tripartite(df_input, graph_type=nx.Graph()):
    df = df_input.copy()
    mapping = {x:node_id for node_id, x in enumerate(set(df.index.values.tolist() + df["cc_num"].values.tolist() +
    df["merchant"].values.tolist()))}
    df["in_node"] = df["cc_num"].apply(lambda x:mapping[x])
    df["out_node"] = df["merchant"].apply(lambda x: mapping[x])

    G = nx.from_edgelist([(x["in_node"], mapping[idx]) for idx, x in df.iterrows()] +
                         [(x["out_node"], mapping[idx]) for idx, x in df.iterrows()],
                         create_using=graph_type)

    nx.set_node_attributes(G, {x["in_node"]: 1 for idx, x in df.iterrows()}, "bipartite")
    nx.set_node_attributes(G, {x["out_node"]: 2 for idx, x in df.iterrows()}, "bipartite")
    nx.set_node_attributes(G, {mapping[idx]: 3 for idx, x in df.iterrows()}, "bipartite")

    nx.set_edge_attributes(G, {(x["in_node"], mapping[idx]): x["is_fraud"] for idx, x in df.iterrows()}, "label")
    nx.set_edge_attributes(G, {(x["out_node"], mapping[idx]): x["is_fraud"] for idx, x in df.iterrows()}, "label")

    nx.set_edge_attributes(G, {(x["in_node"], mapping[idx]): x["amt"] for idx, x in df.iterrows()}, "weight")
    nx.set_edge_attributes(G, {(x["out_node"], mapping[idx]): x["amt"] for idx, x in df.iterrows()}, "weight")

    return G

G_bu = build_graph_bipartite(df, nx.Graph(name='Bipartite Undirected'))
G_bd = build_graph_bipartite(df, nx.DiGraph(name='Bipartite Directed'))
G_tu = build_graph_tripartite(df, nx.Graph(name='Tripartite Undirected'))
G_td = build_graph_tripartite(df, nx.DiGraph(name='Tripartite Directed'))

from networkx.algorithms import bipartite
bipartite.is_bipartite(G_bu)

# print(nx.info(G_bu))
# print(nx.info(G_tu))


for G in [G_bu, G_tu]:
    plt.figure(figsize=(10, 10))
    degrees = pd.Series(
        {
            k: v for k, v in nx.degree(G)
        }
    )
    degrees.plot.hist()
    plt.yscale("log")


for G in [G_bu, G_tu]:
    allEdgesWeights = pd.Series({(d[0], d[1]): d[2]["weight"] for d in G.edges(data=True)})
    np.quantile(allEdgesWeights.values, [0.10, 0.50, 0.70, 0.9, 1.0])
    quant_dist = np.quantile(allEdgesWeights.values, [0.10, 0.50, 0.70, 0.9])
    allEdgesWeightsFiltered = pd.Series({(d[0], d[1]): d[2]["weight"] for d in G.edges(data=True)
                          if d[2]["weight"] < quant_dist[-1]})
    plt.figure(figsize=(10, 10))

    allEdgesWeightsFiltered.plot.hist(bins=40)
    plt.yscale("log")


# nx.degree_pearson_correlation_coefficient(G_bu)
# nx.degree_pearson_correlation_coefficient(G_tu)




# Next part
import community
parts = community.best_partition(G_bu, random_state=42, weight='weight')
communities = pd.Series(parts)
communities.value_counts().sort_values(ascending=False)
hist = communities.value_counts().plot.hist(bins=20)


















plt.show()