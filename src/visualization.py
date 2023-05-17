import algo
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

TOP1_EDGES = [('h1', 's1'),
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


def draw_simulation(edges: list, nlof_scores: list):
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


def test_nlof_case_no_packet_loss():
    np.random.seed(123)

    edges = TOP1_EDGES
    path_vectors = [[] for _ in range(6)]
    tps = [290.21, 290.21, 290.21, 290.21, 290.21, 290.21]
    path_vectors[0] = [0, 9, 11, 6]
    path_vectors[1] = [1, 9, 11, 6]
    path_vectors[2] = [2, 9, 11, 7]

    path_vectors[2] = [3, 10, 11, 7]
    path_vectors[2] = [4, 10, 11, 8]
    path_vectors[2] = [5, 10, 11, 8]

    links = [[] for _ in range(12)]
    for path_i in range(len(path_vectors)):
        tp = tps[path_i]
        for p_ij in path_vectors[path_i]:
            links[p_ij].append(tp)
    nlof_scores = algo.make_simulation(links)
    draw_simulation(edges, nlof_scores)


#       n22,0
#        |
#        |
#  n21,1--n20,2--n23,3
#        \               --------31,10*
#         \             |
#          n00,4---------n30,9-------n32,11
#         /             |
#        /               --------n33,12
#   n13,8--n10,6---n11,5*
#       |
#       n12,7
def test_nlofmll():
    n_links = 12
    tps_1 = [3342.20, 3341.77, 3342.35, 3320.96, 3321.83, 3321.98]
    tps_2 = [3340.77, 3340.70, 3340.56, 3321.47, 3321.76, 3321.69]
    tps_3 = [3340.10, 3340.24, 3339.67, 3321.69, 3321.69, 3321.69]

    path_vectors = [
        [0, 9, 11, 6],
        [1, 9, 11, 6],
        [2, 9, 11, 7],
        [3, 10, 11, 7],
        [4, 10, 11, 8],
        [5, 10, 11, 8]
    ]

    links1, paths = algo.transform_input(tps_1, path_vectors, n_links)
    links2, paths = algo.transform_input(tps_2, path_vectors, n_links)
    links3, paths = algo.transform_input(tps_3, path_vectors, n_links)

    links_batch = [links1, links2, links3]
    paths_batch = [paths, paths, paths]

    wages_batch = algo.learn_nlof_mll(3, links_batch, paths_batch)
    print(wages_batch)
    draw_simulation(TOP1_EDGES, wages_batch[-1])


if __name__ == '__main__':
    test_nlofmll()
    print("finish")
