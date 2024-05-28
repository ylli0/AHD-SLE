import numpy as np
import scipy.sparse as sp
from itertools import combinations

def transform(pairs, v_threshold=10000, e_threshold=10000):
    """construct line expansion from original hypergraph
    INPUT:
        - pairs <matrix>
            - size: N x 2. N means the total vertex-hyperedge pair of the hypergraph
            - each row contains the idx_of_vertex, idx_of_hyperedge
        - v_threshold: vertex-similar neighbor sample threshold
        - e_threshold: hyperedge-similar neighbor sample threshold
    Concept:
        - vertex, hyperedge: for the hypergraph
        - node, edge: for the induced simple graph
    OUTPUT:
        - adj <sparse coo_matrix>: N_node x N_node
        - Pv <sparse coo_matrix>: N_node x N_vertex
        - PvT <sparse coo_matrix>: N_vertex x N_node
        - Pe <sparse coo_matrix>: N_node x N_hyperedge
        - PeT <sparse coo_matrix>: N_hyperedge x N_node
    """
    
    # get # of vertices and encode them starting from 0
    uniq_vertex = np.unique(pairs[:, 0])
    N_vertex = len(uniq_vertex)
    # Adjust the initial number of nodes in pairs to 0
    pairs[:, 0] = list(map({vertex: i for i, vertex in enumerate(uniq_vertex)}.get, pairs[:, 0]))
    
    # get  of hyperedges and encode them starting from 0
    uniq_hyperedge = np.unique(pairs[:, 1])
    N_hyperedge = len(uniq_hyperedge)
    # Adjust the initial number of hyperedge in pairs to 0
    pairs[:, 1] = list(map({hyperedge: i for i, hyperedge in enumerate(uniq_hyperedge)}.get, pairs[:, 1]))

    N_node = pairs.shape[0]

    # vertex projection: from vertex to node
    Pv = sp.coo_matrix((np.ones(N_node), (np.arange(N_node), pairs[:, 0])), 
        shape=(N_node, N_vertex), dtype=np.float32) # (N_node, N_vertex)
    # vertex back projection (Pv Transpose): from node to vertex
    
    e_degree = np.ones(N_hyperedge)
    for vl in range(N_hyperedge):
        tmp = np.where(pairs[:,1] == vl)[0]
        e_degree[vl] = len(tmp)
        
    weight = np.ones(N_node)
    for vertex in range(N_vertex):
        tmp = np.where(pairs[:, 0] == vertex)[0]
        d = e_degree[pairs[tmp,1]]
        s = 0
        for i in d:
            s = s + 1. / i

        weight[tmp] = (1. / d)/s
    PvT = sp.coo_matrix((weight, (pairs[:, 0], np.arange(N_node))), 
        shape=(N_vertex, N_node), dtype=np.float32) # (N_vertex, N_node)
    

    # hyperedge projection: from hyperedge to node
    Pe = sp.coo_matrix((np.ones(N_node), (np.arange(N_node), pairs[:, 1])), 
        shape=(N_node, N_hyperedge), dtype=np.float32) # (N_node, N_hyperedge)
    # hyperedge back projection (Pe Transpose): from node to hyperedge

    v_degree = np.ones(N_vertex)
    for vl in range(N_vertex):
        tmp = np.where(pairs[:,0] == vl)[0]
        v_degree[vl] = len(tmp)

    weight = np.ones(N_node)
    for hyperedge in range(N_hyperedge):
        tmp = np.where(pairs[:, 1] == hyperedge)[0]
        d = v_degree[pairs[tmp,0]]
        s = 0
        for i in d:
            s = s + 1. / i

        weight[tmp] = (1. / d)/s
    PeT = sp.coo_matrix((weight, (pairs[:, 1], np.arange(N_node))), 
        shape=(N_hyperedge, N_node), dtype=np.float32) # (N_node, N_hyperedge)
    
    # construct adj
    edges_v = []
    # get vertex-similar edges
    for vertex in range(N_vertex):
        position = np.where(pairs[:, 0]==vertex)[0]
        if len(position) > v_threshold:
            position = np.random.choice(position, v_threshold, replace=False)
            tmp_edge = np.array(list(combinations(position, r=2)))
            edges_v += list(tmp_edge)
        else:
            edges_v += list(combinations(position, r=2))
    
    edges_e = []
    # get hyperedge-similar edges
    for hyperedge in range(N_hyperedge):
        position = np.where(pairs[:, 1]==hyperedge)[0]
        if len(position) > e_threshold:
            position = np.random.choice(position, e_threshold, replace=False)
            tmp_edge = np.array(list(combinations(position, r=2)))
            edges_e += list(list(tmp_edge))
        else:
            edges_e += list(combinations(position, r=2))

    edges_v = np.array(edges_v)
    edges_e = np.array(edges_e)

    adj_v = sp.coo_matrix((np.ones(edges_v.shape[0]), (edges_v[:, 0], edges_v[:, 1])),
                        shape=(N_node, N_node), dtype=np.float32)
    
    adj_e = sp.coo_matrix((np.ones(edges_e.shape[0]), (edges_e[:, 0], edges_e[:, 1])),
                        shape=(N_node, N_node), dtype=np.float32)

    # print("Pv {}  PvT{}  Pe {}  PeT{}  adj{}".format(Pv.shape,PvT.shape,Pe.shape,PeT.shape,adj_v.shape))

    return adj_v,adj_e, Pv, PvT, Pe, PeT
