import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm


def sparse_solve_gmres(a, b, precondition=False):
    """Solve linear equations a.x = b for x using generalized minimal residual method."""
    precon = spla.LinearOperator(a.shape, spla.spilu(a).solve) if precondition else None
    x, info = spla.gmres(a, b, M=precon)
    print(f'GMRES solver finished with exit code {info}')
    return x


def process_graph(file_location):
    graph = nx.read_gpickle(file_location)
    print('Graph successfully read from file')
    nodelist = list(graph)
    neighbour_count = [len(graph.in_edges(n)) + len(graph.out_edges(n)) for n in nodelist]
    adj_mat = nx.to_scipy_sparse_matrix(graph, nodelist=nodelist)
    print('Adjacency matrix calculated from graph')
    directed_edge_set = set(graph.out_edges)
    edge_set = directed_edge_set.union({e[::-1] for e in directed_edge_set})
    print('Graph edge set generated')
    return edge_set, nodelist, neighbour_count, adj_mat


def linear_system_row(edge_set, nodelist, neighbour_count, num_nodes, row_index):
    row = np.ones(num_nodes, dtype=int)
    zero_indices = [j for j in range(num_nodes) if (nodelist[row_index], nodelist[j]) in edge_set]
    row[zero_indices] = 0
    row[row_index] = neighbour_count[row_index] + 1
    return row


def linear_system(edge_set, nodelist, neighbour_count, adj_mat):
    num_nodes = len(nodelist)
    max_abs_weight = max(adj_mat.max(), - adj_mat.min())
    system_vec = adj_mat.sum(axis=1).A1 - adj_mat.sum(axis=0).A1  # A - A^T
    system_vec += max_abs_weight * (num_nodes - 1)
    print('Initial linear system vector calculated')
    min_neighbour_idx = np.argmin(neighbour_count)
    min_neighbour_row = linear_system_row(edge_set, nodelist, neighbour_count, num_nodes, min_neighbour_idx)
    system_mat = sp.lil_matrix((num_nodes, num_nodes), dtype=int)
    system_mat[min_neighbour_idx] = min_neighbour_row
    for i in tqdm(range(num_nodes), total=num_nodes, desc='Calculating reduced linear system'):
        if i != min_neighbour_idx:
            system_vec[i] = system_vec[i] - system_vec[min_neighbour_idx]
            row = linear_system_row(edge_set, nodelist, neighbour_count, num_nodes, i) - min_neighbour_row
            system_mat[i] = row
    system_mat = system_mat.tocsr()
    print('Linear system prepared to solve')
    return system_mat, system_vec


def map_difficulties(file_location):
    edge_set, nodelist, neighbour_count, adj_mat = process_graph(file_location)
    system_mat, system_vec = linear_system(edge_set, nodelist, neighbour_count, adj_mat)
    diffs = sparse_solve_gmres(system_mat, system_vec)
    return nodelist, diffs


if __name__ == '__main__':
    filename = 'comparison_graph.gpickle'
    node_list, difficulties = map_difficulties(filename)
    difficulties /= np.amax(difficulties)
