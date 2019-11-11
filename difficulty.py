import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def sparse_solve_gmres(a, b, precondition=False):
    """Solve linear equations a.x = b for x using generalized minimal residual method."""
    precon = spla.LinearOperator(a.shape, spla.spilu(a).solve) if precondition else None
    x, info = spla.gmres(a, b, M=precon)
    print(f'GMRES solver finished with exit code {info}')
    return x


def process_graph(file_location):
    graph = nx.read_gpickle(file_location)
    nodelist = list(graph)
    neighbour_count = np.array([len(list(graph.predecessors(n))) + len(list(graph.successors(n))) for n in nodelist])
    adj_mat = nx.to_scipy_sparse_matrix(graph, nodelist=nodelist)
    undirected_graph = graph.to_undirected(as_view=True)
    return undirected_graph, nodelist, neighbour_count, adj_mat


def linear_system_row(undirected_graph, nodelist, neighbour_count, num_nodes, row_index):
    row = np.ones(num_nodes, dtype=int)
    zero_indices = [j for j in range(num_nodes) if undirected_graph.has_edge(nodelist[row_index], nodelist[j])]
    row[zero_indices] = 0
    row[row_index] = neighbour_count[row_index] + 1
    return row


def linear_system(undirected_graph, nodelist, neighbour_count, adj_mat):
    shape = adj_mat.shape
    comp_mat = adj_mat - adj_mat.transpose()
    num_nodes = shape[0]
    system_vec = 1 + comp_mat.sum(axis=1) / (comp_mat.max() * (num_nodes - 1))
    system_mat = sp.lil_matrix(shape, dtype=int)
    min_neighbour_idx = neighbour_count.argmin()
    min_neighbour_row = linear_system_row(undirected_graph, nodelist, neighbour_count, num_nodes, min_neighbour_idx)
    system_mat[min_neighbour_idx] = min_neighbour_row
    for i in range(num_nodes):
        if i != min_neighbour_idx:
            row = linear_system_row(undirected_graph, nodelist, neighbour_count, num_nodes, i)
            system_mat[i] = min_neighbour_row - row
            system_vec[i] = system_vec[min_neighbour_idx] - system_vec[i]
    return system_mat.tocsr(), system_vec


def map_difficulties(file_location):
    undirected_graph, nodelist, neighbour_count, adj_mat = process_graph(file_location)
    system_mat, system_vec = linear_system(undirected_graph, nodelist, neighbour_count, adj_mat)
    diffs = sparse_solve_gmres(system_mat, system_vec)
    return nodelist, diffs


if __name__ == '__main__':
    filename = 'comparison_graph.gpickle'
    node_list, difficulties = map_difficulties(filename)
    difficulties /= np.amax(difficulties)
