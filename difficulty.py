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
    print('Graph successfully read from file')
    nodelist = list(graph)
    num_nodes = len(nodelist)
    max_abs_weight = max(abs(e[-1]['weight']) for e in graph.out_edges(data=True))
    print('Found maximum weight magnitude of edges')
    neighbour_count = []
    system_vec = []
    for n in nodelist:
        out_edges = graph.out_edges(n, data=True)
        in_edges = graph.in_edges(n, data=True)
        neighbour_count.append(len(in_edges) + len(out_edges))
        vec_component = sum(e[-1]['weight'] for e in out_edges)
        vec_component -= sum(e[-1]['weight'] for e in in_edges)
        vec_component += max_abs_weight * (num_nodes - 1)
        system_vec.append(vec_component)
    print('Initial linear system vector calculated')
    undirected_graph = graph.to_undirected(as_view=True)
    return undirected_graph, nodelist, neighbour_count, num_nodes, system_vec


def linear_system_row(undirected_graph, nodelist, neighbour_count, num_nodes, row_index):
    row = np.ones(num_nodes, dtype=int)
    zero_indices = [j for j in range(num_nodes) if undirected_graph.has_edge(nodelist[row_index], nodelist[j])]
    row[zero_indices] = 0
    row[row_index] = neighbour_count[row_index] + 1
    return row


def linear_system(undirected_graph, nodelist, neighbour_count, num_nodes, system_vec):
    min_neighbour_idx = neighbour_count.index(min(neighbour_count))
    min_neighbour_row = linear_system_row(undirected_graph, nodelist, neighbour_count, num_nodes, min_neighbour_idx)
    system_mat = sp.lil_matrix((num_nodes, num_nodes), dtype=int)
    system_mat[min_neighbour_idx] = min_neighbour_row
    for i in range(num_nodes):
        if i != min_neighbour_idx:
            row = linear_system_row(undirected_graph, nodelist, neighbour_count, num_nodes, i)
            system_mat[i] = min_neighbour_row - row
            system_vec[i] = system_vec[min_neighbour_idx] - system_vec[i]
    return system_mat.tocsr(), system_vec


def map_difficulties(file_location):
    undirected_graph, nodelist, neighbour_count, num_nodes, system_vec = process_graph(file_location)
    system_mat, system_vec = linear_system(undirected_graph, nodelist, neighbour_count, num_nodes, system_vec)
    print('Linear system prepared to solve')
    diffs = sparse_solve_gmres(system_mat, system_vec)
    return nodelist, diffs


if __name__ == '__main__':
    filename = 'comparison_graph.gpickle'
    node_list, difficulties = map_difficulties(filename)
    difficulties /= np.amax(difficulties)
