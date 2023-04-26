import algo
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_network(edges, edge_labels):
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(
        G, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9,
        labels={node: node for node in G.nodes()}
    )
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='red'
    )
    plt.axis('off')
    plt.show()


def draw_simulation(edges: list, links: list, nlof_scores: list):
    nlof_scores = [round(x, 4) for x in nlof_scores]
    edge_labels = dict(zip(edges, nlof_scores))
    draw_network(edges, edge_labels)


def test_case_simple():
    edges = [(0, 1), (1, 2), (2, 3)]
    links = [[19, 15, 17, 20],
             [22, 21.5, 20, 18],
             [20, 20.5, 20.3, 20.5]]
    nlof_scores = algo.make_simulation(links)
    draw_simulation(edges, links, nlof_scores)


def test_case_presentation():
    np.random.seed(123)
    edges = [('h1', 's1'),
             ('h2', 's1'),
             ('h3', 's1'),
             ('h4', 's2'),
             ('h5', 's2'),
             ('h6', 's2'),
             ('h7', 's3'),
             ('h8', 's3'),
             ('h9', 's3'),
             ('s1', 's4'),
             ('s2', 's4'),
             ('s3', 's4')]
    path_vectors = [[] for _ in range(3)]
    tps = [10.5, 10.2, 20]
    path_vectors[0] = [0, 9, 11, 6]
    path_vectors[1] = [1, 9, 11, 6]
    path_vectors[2] = [3, 10, 9, 2]

    links = [[] for _ in range(12)]
    for i in range(len(path_vectors)):
        tp = tps[i]
        for p_ij in path_vectors[i]:
            tp_real = np.random.normal(tp, 0.5, 1).tolist()[0]
            print(tp_real)
            links[p_ij].append(tp_real)
    nlof_scores = algo.make_simulation(links)
    draw_simulation(edges, links, nlof_scores)


if __name__ == '__main__':
    test_case_presentation()
