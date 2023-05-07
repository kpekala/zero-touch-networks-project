import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from math import inf

from src.utils import flatten


def make_simulation(links):
    flows = flatten(links)

    # Stage 1
    base_clusters, noise = dbscan(np.array(flows))
    tp_clusters = tp_cluster(base_clusters, noise, 0.5, 1)

    # Stage 2 and 3
    f, trace = compute_fof(tp_clusters)

    # Stage 4
    nlof_scores = compute_nlof_scores(links, trace, 0.3)

    return nlof_scores


def dbscan(x):
    x_list = x.tolist()
    clustering = DBSCAN(eps=2).fit(x.reshape(-1, 1))
    x_labels = clustering.labels_.tolist()
    clusters = [[] for _ in range(max(x_labels) + 1)]
    noise = []
    for i in range(len(x_list)):
        if x_labels[i] == -1:
            noise.append(x_list[i])
        else:
            clusters[x_labels[i]].append(x_list[i])
    return clusters, noise


def tp_cluster(clusters: list, noise: list, tp_ratio: float, tp_deviation: float):
    """
        Parameters :
            clusters : list of DBSCAN cluster sets in descending throughput order
            noise : list of DBSCAN noise flows
            tp_ratio : float
                Ratio used to determine if two DBSCAN clusters can be combined into one TPCluster
            tp_deviation : float
                The relative distance a noise flow can be away from a TPCluster to be assigned to that cluster

        Returns :
            cs : set of TPClusters
    """

    r = 0
    cs = []
    for cluster in clusters:
        if not subset_of_sets(cluster, cs):
            cs.append(cluster)
            m = max(cluster)
            for cluster_k in clusters:
                if cluster_k is cluster: continue
                m_prim = max(cluster_k)
                if (1 - tp_ratio) * m < m_prim < m:
                    cs[r] += cluster_k
            r += 1
    for n_j in noise:
        delta_min = inf
        a = None
        for i in range(len(cs)):
            m = max(cs[i])
            if (-tp_deviation * m) <= (m - n_j) <= delta_min:
                delta_min = m - n_j
                a = i
        if a:
            cs[a].append(n_j)
        else:
            cs[0].append(n_j)
    return cs


def subset_of_sets(_set: set, sets):
    return len([s for s in sets if all([x in s for x in _set])]) > 0


def compute_fof(cs):
    """
        Parameters:
            cs : list of TPClusters

        Returns:
            f : list
                FOF score for each flow in each cluster
            trace : dict
                dictionary mapping throughput to fof score

    """
    f = [[_ for _ in c_i] for c_i in cs]
    trace = dict()
    k = len(cs)
    for i in range(k):
        c_np = np.array(cs[i])
        s_labels = KMeans(n_clusters=k).fit(c_np.reshape((-1, 1))).labels_
        s_count = max(s_labels) + 1
        c_prim = 0
        for s_i in range(s_count):
            c_prim = max(c_prim, np.sum(c_np * (s_labels == s_i)) / np.sum(s_labels == s_i))
        for j in range(len(cs[i])):
            f[i][j] = np.abs(cs[i][j] - c_prim) / np.abs(c_prim)
            trace[cs[i][j]] = f[i][j]
    return f, trace


def compute_nlof_scores(links: list, trace: dict, gamma: float):
    """
        Parameters:
            links : 2d list
                links, each link contains list of flows (throughput)
            trace : dict
                dictionary mapping throughput to fof score
            gamma : float
                Outlier threshold

        Returns:
            scores : list
                nlof scores for each link
    """
    scores = [0.0] * len(links)
    for i in range(len(links)):
        if len(links[i]) == 0:
            continue
        r = 0
        for j in range(len(links[i])):
            if trace[links[i][j]] > gamma:
                r += 1
        scores[i] = r / len(links[i])
    return scores


def compute_nlof_mll_scores(links_batches: list, path_batches: list, path_trace: dict, trace: dict, gamma: float):
    N = len(links_batches[0])  # number of links
    T = len(links_batches)  # number of epochs
    wages = [[0.0] * N for _ in range(T)]
    for t in range(T):
        paths = [path_trace[path_batches[t][i]] for i in range(N)]
        update_wages_for_epoch(wages[:t], links_batches[t], paths, trace, gamma, t)

    return wages[-1]


def learn_nlof_mll(epochs_numb: int, flows_batch: list[list], paths_batch: list[list]):
    gamma = 0.3
    wages_batch = [[] for _ in range(epochs_numb)]
    for epoch in range(epochs_numb):
        flows = flows_batch[epoch]
        paths = paths_batch[epoch]

        base_clusters, noise = dbscan(np.array(flatten(flows)))
        tp_clusters = tp_cluster(base_clusters, noise, 0.5, 1)

        f, trace = compute_fof(tp_clusters)

        wages_batch[epoch] = update_wages_for_epoch(wages_batch[:epoch], flows, paths, trace, gamma, epoch)
    return wages_batch


def update_wages_for_epoch(prev_wages: list, flows: list, paths: list, trace: dict, gamma: float, t: int):
    EPS = 0.001  # small initial value for every wage
    delta = 0.1  # learning speed
    K = 1  # number of previous epochs to learn from
    N = len(flows)

    if t == 0:
        initial_wages = [EPS] * N
    else:
        initial_wages = [sum([prev_wages[-j][i] for j in range(1, K + 1)]) / K for i in range(N)]
    wages = initial_wages
    for i in range(N):
        for f in range(len(flows[i])):
            fof = trace[flows[i][f]]
            if fof > gamma:
                p = paths[i][f]
                wages[i] += wages[i] / np.dot(np.array(wages), np.array(p))
            else:
                wages[i] -= min(1.0, delta * wages[i])

        # normalise wages
        wages[i] = wages[i] / (len(flows[i]) + initial_wages[i])
    return wages
