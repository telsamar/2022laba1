# Step 1: Load the libraries
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from node2vec.edges import HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder
from node2vec import Node2Vec  # pip install node2vec, pip install python-Levenshtein
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# %matplotlib inline # For Jupyter Notebook
from networkx.algorithms import bipartite
import scipy as sp
import scipy.stats
import community  # pip install python-louvain

# Step 2: Load the dataset
data = pd.read_csv('data/fraudTrain.csv')
df1 = data[data["is_fraud"] == 0].sample(frac=0.20, random_state=42)
df2 = data[data["is_fraud"] == 1]
df = pd.concat([df1, df2])
# df.head() # For Jupyter Notebook

# Step 3: Data Intuition
# df.info() # For Jupyter Notebook

# Step 4: Target Variable Distribution
# df["is_fraud"].value_counts() # For Jupyter Notebook

# Data Preparation — Creating Graph Nodes and Edges from Dataframe


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
    return G
    # plt.figure(figsize=(7, 4))
    # plt.show()


def build_graph_tripartite(df_input, graph_type=nx.Graph()):
    df = df_input.copy()
    mapping = {x: node_id for node_id, x in enumerate(set(df.index.values.tolist() + df["cc_num"].values.tolist() +
                                                          df["merchant"].values.tolist()))}
    df["in_node"] = df["cc_num"].apply(lambda x: mapping[x])
    df["out_node"] = df["merchant"].apply(lambda x: mapping[x])

    G = nx.from_edgelist([(x["in_node"], mapping[idx]) for idx, x in df.iterrows()] +
                         [(x["out_node"], mapping[idx])
                          for idx, x in df.iterrows()],
                         create_using=graph_type)

    nx.set_node_attributes(
        G, {x["in_node"]: 1 for idx, x in df.iterrows()}, "bipartite")
    nx.set_node_attributes(
        G, {x["out_node"]: 2 for idx, x in df.iterrows()}, "bipartite")
    nx.set_node_attributes(
        G, {mapping[idx]: 3 for idx, x in df.iterrows()}, "bipartite")

    nx.set_edge_attributes(
        G, {(x["in_node"], mapping[idx]): x["is_fraud"] for idx, x in df.iterrows()}, "label")
    nx.set_edge_attributes(
        G, {(x["out_node"], mapping[idx]): x["is_fraud"] for idx, x in df.iterrows()}, "label")

    nx.set_edge_attributes(G, {(x["in_node"], mapping[idx]): x["amt"]
                           for idx, x in df.iterrows()}, "weight")
    nx.set_edge_attributes(G, {(x["out_node"], mapping[idx]): x["amt"]
                           for idx, x in df.iterrows()}, "weight")
    return G
    # plt.figure(figsize=(7, 4))
    # plt.show()


G_bu = build_graph_bipartite(df, nx.Graph(name='Bipartite Undirected'))
G_bd = build_graph_bipartite(df, nx.DiGraph(name='Bipartite Directed'))
G_tu = build_graph_tripartite(df, nx.Graph(name='Tripartite Undirected'))
G_td = build_graph_tripartite(df, nx.DiGraph(name='Tripartite Directed'))


bipartite.is_bipartite(G_bu)

# Network Topology - Statistical Analysis
# print(nx.info(G_bu)) # For Jupyter Notebook, Warning: info() will be deprecated soon
# print(nx.info(G_tu)) # For Jupyter Notebook, Warning: info() will be deprecated soon

# Nodes Degree Distribution
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

# Edge Weight Distribution
for G in [G_bu, G_tu]:
    allEdgesWeights = pd.Series(
        {(d[0], d[1]): d[2]["weight"] for d in G.edges(data=True)})
    np.quantile(allEdgesWeights.values, [0.10, 0.50, 0.70, 0.9, 1.0])
    quant_dist = np.quantile(allEdgesWeights.values, [0.10, 0.50, 0.70, 0.9])
    allEdgesWeightsFiltered = pd.Series({(d[0], d[1]): d[2]["weight"] for d in G.edges(data=True)
                                         if d[2]["weight"] < quant_dist[-1]})
    plt.figure(figsize=(10, 10))

    allEdgesWeightsFiltered.plot.hist(bins=40)
    plt.yscale("log")
#     # plt.figure(figsize=(7, 4))
#     # plt.show()

# Assortativity
nx.degree_pearson_correlation_coefficient(G_bu)  # For Jupyter Notebook
nx.degree_pearson_correlation_coefficient(G_tu)  # For Jupyter Notebook


# Community Detection & Fraudulent Edges Ratio Analysis
parts = community.best_partition(G_bu, random_state=42, weight='weight')
communities = pd.Series(parts)
print(communities.value_counts().sort_values(ascending=False))

# Communities List from the Graph
communities.value_counts().plot.hist(bins=20)  # For Jupyter Notebook
# plt.figure(figsize=(8, 4))
# plt.show()


# Следующий код генерирует индуцированный узлом подграф с использованием узлов, присутствующих в конкретном сообществе.
graphs = []
d = {}
for x in communities.unique():
    tmp = nx.subgraph(G_bu, communities[communities == x].index)
    fraud_edges = sum(nx.get_edge_attributes(tmp, "label").values())
    ratio = 0 if fraud_edges == 0 else (
        fraud_edges / tmp.number_of_edges()) * 100
    d[x] = ratio
    graphs += [tmp]

pd.Series(d).sort_values(ascending=False)  # For Jupyter Notebook

# Мы можем пойти еще на один шаг вперед и построить индуцированные узлами подграфы для конкретного сообщества

gId = 1  # можно менять для разных сообществ
plt.figure(figsize=(10, 10))
spring_pos = nx.spring_layout(graphs[gId])
plt.axis("off")
edge_colors = ["r" if x == 1 else "g" for x in nx.get_edge_attributes(
    graphs[gId], 'label').values()]
nx.draw_networkx(graphs[gId], pos=spring_pos,
                 edge_color=edge_colors, with_labels=True, node_size=15)
plt.show()

# # Доброкачественные края → Представлены зеленым
# # Мошеннические края → Представлено красным

# Handling Class Imbalance
df_majority = df[df.is_fraud == 0]
df_minority = df[df.is_fraud == 1]

df_maj_downsampled = resample(df_majority,
                              n_samples=len(df_minority),
                              random_state=42)

df_downsampled = pd.concat([df_minority, df_maj_downsampled])

print(df_downsampled.is_fraud.value_counts())
G_down = build_graph_bipartite(df_downsampled)

# Train Test Split

train_edges, test_edges, train_labels, test_labels = train_test_split(list(range(len(G_down.edges))),
                                                                      list(nx.get_edge_attributes(G_down,
                                                                                                  "label").values()),
                                                                      test_size=0.20, random_state=42)

# Graph Embeddings
edgs = list(G_down.edges)
train_graph = G_down.edge_subgraph([edgs[x] for x in train_edges]).copy()
train_graph.add_nodes_from(list(set(G_down.nodes) - set(train_graph.nodes)))
node2vec_train = Node2Vec(train_graph, weight_key='weight')
model_train = node2vec_train.fit(window=10)

# Supervised Model Building

classes = [HadamardEmbedder, AverageEmbedder,
           WeightedL1Embedder, WeightedL2Embedder]
for cl in classes:
    embeddings_train = cl(keyed_vectors=model_train.wv)
    train_embeddings = [embeddings_train[str(
        edgs[x][0]), str(edgs[x][1])] for x in train_edges]
    test_embeddings = [embeddings_train[str(
        edgs[x][0]), str(edgs[x][1])] for x in test_edges]

    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(train_embeddings, train_labels)

    y_pred = rf.predict(test_embeddings)
    print(cl)
    print('Precision:', metrics.precision_score(test_labels, y_pred))
    print('Recall:', metrics.recall_score(test_labels, y_pred))
    print('F1-Score:', metrics.f1_score(test_labels, y_pred))


# Model Building — Unsupervised Learning
nod2vec_unsup = Node2Vec(G_down, weight_key='weight')
unsup_vals = nod2vec_unsup.fit(window=10)


classes = [HadamardEmbedder, AverageEmbedder,
           WeightedL1Embedder, WeightedL2Embedder]
true_labels = [x for x in nx.get_edge_attributes(G_down, "label").values()]

for cl in classes:
    embedding_edge = cl(keyed_vectors=unsup_vals.wv)

    embedding = [embedding_edge[str(x[0]), str(x[1])] for x in G_down.edges()]
    kmeans = KMeans(2, random_state=42).fit(embedding)

    nmi = metrics.adjusted_mutual_info_score(true_labels, kmeans.labels_)
    ho = metrics.homogeneity_score(true_labels, kmeans.labels_)
    co = metrics.completeness_score(true_labels, kmeans.labels_)
    vmeasure = metrics.v_measure_score(true_labels, kmeans.labels_)

    print(cl)
    print('NMI:', nmi)
    print('Homogeneity:', ho)
    print('Completeness:', co)
    print('V-Measure:', vmeasure)
