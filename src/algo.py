import numpy as np
import tensorflow as tf
from sklearn.cluster import DBSCAN, KMeans
from matplotlib import pyplot as plt
from math import inf


def dbscan(x):
    x_list = x.tolist()
    clustering = DBSCAN(eps=2).fit(x.reshape(-1, 1))
    x_labels = clustering.labels_.tolist()
    clusters = [[] for _ in range(max(x_labels) + 1)]
    noice = []
    for i in range(len(x_list)):
        if x_labels[i] == -1:
            noice.append(x_list[i])
        else:
            clusters[x_labels[i]].append(x_list[i])
    return clusters, noice


def subset_of_sets(_set: set, sets):
    return len([s for s in sets if all([x in s for x in _set])]) > 0


def tp_cluster(clusters: set, noise: set, tp_ratio: float, tp_deviation: float):
    """
        Parameters :
            clusters : set
                Set of DBSCAN cluster sets in descending throughput order
            noise : set
                Set of DBSCAN noice flows
            tp_ratio : float
                Ratio used to determine if two DBSCAN clusters can be combined into one TPCluster
            tp_deviation : float
                The relative distance a noise flow can be away from a TPCluster to be assigned to that cluster

        Returns :
            cs : set
                set of TPClusters
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


def compute_fof(cs):
    """
        Parameters:
            cs : list of TPClusters

        Returns:
            f: 2d list
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


def compute_nlof_scores(links, trace, gamma):
    """
        Parameters:
            links : 2d list
                links, each link contains list of flows (throughput)
            trace : dict
                dictionary mapping throughput to fof score
            gamma : float
                Outlier treshold

        Returns:
            scores : list
                nlof scores for each link
    """
    scores = [0.0] * len(links)
    for i in range(len(links)):
        r = 0
        for j in range(len(links[i])):
            if trace[links[i][j]] > gamma:
                r += 1
        scores[i] = r / len(links[i])
    return scores
