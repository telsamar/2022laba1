import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt




data = pd.read_csv('data/fraudTrain.csv')
# print(data.head())
df1 = data[data["is_fraud"] == 0].sample(frac=0.20, random_state=42)
# print(df1.head())
df2 = data[data["is_fraud"] == 1]
# print(df2.head())
df = pd.concat([df1, df2])
# print(df.head())

print(df["is_fraud"].value_counts())


def build_graph_bipartite(df_input, graph_type=nx.Graph()):
    df = df_input.copy()
    mapping = {x: node_id for node_id, x in enumerate(set(df["cc_num"].values.tolist()
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

    # plt.figure(figsize=(7, 4))
    # plt.show()

    return G


def build_graph_tripartite(df_input, graph_type=nx.Graph()):
    df = df_input.copy()
    mapping = {x: node_id for node_id, x in enumerate(set(df.index.values.tolist() + df["cc_num"].values.tolist() +
                                                          df["merchant"].values.tolist()))}
    df["in_node"] = df["cc_num"].apply(lambda x: mapping[x])
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

    # plt.figure(figsize=(7, 4))
    # plt.show()

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
    # plt.figure(figsize=(7, 4))
    # plt.show()

for G in [G_bu, G_tu]:
    allEdgesWeights = pd.Series({(d[0], d[1]): d[2]["weight"] for d in G.edges(data=True)})
    np.quantile(allEdgesWeights.values, [0.10, 0.50, 0.70, 0.9, 1.0])
    quant_dist = np.quantile(allEdgesWeights.values, [0.10, 0.50, 0.70, 0.9])
    allEdgesWeightsFiltered = pd.Series({(d[0], d[1]): d[2]["weight"] for d in G.edges(data=True)
                                         if d[2]["weight"] < quant_dist[-1]})
    plt.figure(figsize=(10, 10))

    allEdgesWeightsFiltered.plot.hist(bins=40)
    plt.yscale("log")
    # plt.figure(figsize=(7, 4))
    # plt.show()

# nx.degree_pearson_correlation_coefficient(G_bu)
# nx.degree_pearson_correlation_coefficient(G_tu)


# Next part
import community

parts = community.best_partition(G_bu, random_state=42, weight='weight')
communities = pd.Series(parts)
print(communities.value_counts().sort_values(ascending=False))

communities.value_counts().plot.hist(bins=20)
plt.figure(figsize=(8,4))
# plt.show()


# Следующий код генерирует индуцированный узлом подграф с использованием узлов, присутствующих в конкретном сообществе.
graphs = []
d ={}
for x in communities.unique():
    tmp = nx.subgraph(G_bu, communities[communities == x].index)
    fraud_edges = sum(nx.get_edge_attributes(tmp, "label").values())
    ratio = 0 if fraud_edges == 0 else (fraud_edges / tmp.number_of_edges()) * 100
    d[x] = ratio
    graphs += [tmp]

pd.Series(d).sort_values(ascending=False)


# Мы можем пойти еще на один шаг вперед и построить индуцированные узлами подграфы для конкретного сообщества

gId = 1 #можно менять для разных сообществ
plt.figure(figsize=(10, 10))
spring_pos = nx.spring_layout(graphs[gId])
plt.axis("off")
edge_colors = ["r" if x == 1 else "g" for x in nx.get_edge_attributes(graphs[gId], 'label').values()]
nx.draw_networkx(graphs[gId], post=spring_pos, node_color=0,
                 edge_color=1, with_labels=True, node_size=15)
plt.show()

# Доброкачественные края → Представлены зеленым
# Мошеннические края → Представлено красным

