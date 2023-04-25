import algo
import matplotlib.pyplot as plt
import networkx as nx


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
    edge_labels = dict(zip(edges, nlof_scores))
    draw_network(edges, edge_labels)


if __name__ == '__main__':
    edges = [(0, 1), (1, 2), (2, 3)]
    links = [[19, 15, 17, 20],
             [22, 21.5, 20, 18],
             [20, 20.5, 20.3, 20.5]]
    nlof_scores = algo.make_simulation(links)
    print(nlof_scores)
    draw_simulation(edges, links, nlof_scores)
