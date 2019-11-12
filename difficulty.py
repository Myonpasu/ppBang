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
    undirected_graph = graph.to_undirected(as_view=True)
    return undirected_graph, nodelist, neighbour_count, adj_mat


def linear_system_row(undirected_graph, nodelist, neighbour_count, num_nodes, row_index):
    row = np.ones(num_nodes, dtype=int)
    zero_indices = [j for j in range(num_nodes) if undirected_graph.has_edge(nodelist[row_index], nodelist[j])]
    row[zero_indices] = 0
    row[row_index] = neighbour_count[row_index] + 1
    return row


def row_to_ijv(row_data, row_length, row_index):
    col = []
    data = []
    for j in range(row_length):
        row_element = row_data[j]
        if row_element != 0:
            col.append(j)
            data.append(row_element)
    row = [row_index] * len(col)
    return row, col, data


def linear_system(undirected_graph, nodelist, neighbour_count, adj_mat):
    num_nodes = len(nodelist)
    max_abs_weight = max(adj_mat.max(), - adj_mat.min())
    system_vec = adj_mat.sum(axis=1).A1 - adj_mat.sum(axis=0).A1  # A - A^T
    system_vec += max_abs_weight * (num_nodes - 1)
    print('Initial linear system vector calculated')
    min_neighbour_idx = neighbour_count.index(min(neighbour_count))
    min_neighbour_row = linear_system_row(undirected_graph, nodelist, neighbour_count, num_nodes, min_neighbour_idx)
    row_ind, col_ind, data = row_to_ijv(min_neighbour_row, num_nodes, min_neighbour_idx)
    for i in tqdm(range(num_nodes), total=num_nodes, desc='Calculating reduced linear system'):
        if i != min_neighbour_idx:
            system_vec[i] = system_vec[i] - system_vec[min_neighbour_idx]
            row = linear_system_row(undirected_graph, nodelist, neighbour_count, num_nodes, i) - min_neighbour_row
            new_row_ind, new_col_ind, new_data = row_to_ijv(row, num_nodes, i)
            row_ind.extend(new_row_ind)
            col_ind.extend(new_col_ind)
            data.extend(new_data)
    system_mat = sp.csr_matrix((data, (row_ind, col_ind)), shape=(num_nodes, num_nodes))
    print('Linear system prepared to solve')
    return system_mat, system_vec


def map_difficulties(file_location):
    undirected_graph, nodelist, neighbour_count, adj_mat = process_graph(file_location)
    system_mat, system_vec = linear_system(undirected_graph, nodelist, neighbour_count, adj_mat)
    diffs = sparse_solve_gmres(system_mat, system_vec)
    return nodelist, diffs


if __name__ == '__main__':
    filename = 'comparison_graph.gpickle'
    node_list, difficulties = map_difficulties(filename)
    difficulties /= np.amax(difficulties)
