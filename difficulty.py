import graph_tool as gt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from graph_tool.spectral import adjacency
from tqdm import tqdm


def adjacency_matrix(graph):
    adj_mat = adjacency(graph, weight=graph.ep.weight)
    print('Adjacency matrix calculated from graph')
    return adj_mat


def difficulty_dict(nodelist, diffs):
    diff_dict = {node: diff for node, diff in zip(nodelist, diffs)}
    return diff_dict


def difficulty_output(graph_filename):
    nodelist, diffs = map_difficulties(graph_filename)
    diffs = normalize_difficulties(diffs)
    return difficulty_dict(nodelist, diffs)


def linear_system(graph):
    neighbour_counts = vertex_degrees(graph)
    vertex_count = graph.num_vertices()
    system_vec = system_vector(graph, vertex_count)
    print('Initial linear system vector calculated')
    min_neighbour_idx = np.argmin(neighbour_counts)
    min_neighbour_row = linear_system_row(graph, neighbour_counts, vertex_count, min_neighbour_idx)
    system_mat = sp.lil_matrix((vertex_count, vertex_count), dtype=int)
    system_mat[min_neighbour_idx] = min_neighbour_row
    for i in tqdm(range(vertex_count), total=vertex_count, desc='Calculating reduced linear system'):
        if i != min_neighbour_idx:
            system_vec[i] = system_vec[i] - system_vec[min_neighbour_idx]
            row = linear_system_row(graph, neighbour_counts, vertex_count, i) - min_neighbour_row
            system_mat[i] = row
    system_mat = system_mat.tocsr()
    print('Linear system prepared to solve')
    return system_mat, system_vec


def linear_system_row(graph, neighbour_counts, vertex_count, row_index):
    row = np.ones(vertex_count, dtype=int)
    zero_indices = [graph.vertex_index[v] for v in graph.vertex(row_index).all_neighbors()]  # Not necessarily sorted.
    row[zero_indices] = 0
    row[row_index] = neighbour_counts[row_index] + 1
    return row


def map_difficulties(file_location):
    graph = read_graph(file_location)
    names = vertex_names(graph)
    system_mat, system_vec = linear_system(graph)
    diffs = sparse_solve_gmres(system_mat, system_vec)
    return names, diffs


def normalize_difficulties(diffs):
    diffs /= np.amax(diffs)
    return diffs


def read_graph(file_location):
    graph = gt.load_graph(file_location, fmt='gt')
    print('Graph successfully read from file')
    return graph


def sparse_solve_gmres(a, b, precondition=False):
    """Solve linear equations a.x = b for x using generalized minimal residual method."""
    precon = spla.LinearOperator(a.shape, spla.spilu(a).solve) if precondition else None
    x, info = spla.gmres(a, b, M=precon)
    print(f'GMRES solver finished with exit code {info}')
    return x


def system_vector(graph, vertex_count):
    adj_mat = adjacency_matrix(graph)
    max_abs_weight = max(graph.ep.weight.a.max(), - graph.ep.weight.a.min())
    system_vec = adj_mat.sum(axis=1).A1 - adj_mat.sum(axis=0).A1  # Sum (A - A^T) matrix over its columns.
    system_vec += max_abs_weight * (vertex_count - 1)
    print('Initial linear system vector calculated')
    return system_vec


def vertex_degrees(graph):
    degrees = graph.get_total_degrees(graph.get_vertices())
    print('Vertex degrees calculated from graph')
    return degrees


def vertex_names(graph):
    names = np.asarray(list(graph.vp.name))
    print('Vertex names retrieved from graph')
    return names


if __name__ == '__main__':
    filename = 'comparison_graph.gpickle'
    node_list, difficulties = map_difficulties(filename)
    difficulties /= np.amax(difficulties)
