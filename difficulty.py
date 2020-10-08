import graph_tool as gt
import numpy as np
from tqdm import tqdm


def difficulty_dict(nodelist, diffs):
    diff_dict = {node: diff for node, diff in zip(nodelist, diffs)}
    return diff_dict


def difficulty_output(graph_filename):
    nodelist, diffs = map_difficulties(graph_filename)
    diffs = normalize_difficulties(diffs)
    return difficulty_dict(nodelist, diffs)


def diffs_iterative(graph, epsilon=1e-4):
    vertex_count = graph.num_vertices()
    max_weight = max(graph.ep.weight_accuracy.a.max(), - graph.ep.weight_accuracy.a.min())
    k = vertex_degrees(graph)
    weight_in = gt.incident_edges_op(graph, 'in', 'sum', graph.ep.weight_accuracy)
    weight_out = gt.incident_edges_op(graph, 'out', 'sum', graph.ep.weight_accuracy)
    weight_net = weight_in.a - weight_out.a
    weight_term = weight_net / (max_weight * (vertex_count - 1))
    diffs = graph.new_vp('double', val=(1 / vertex_count))
    neighbor_diff_sum = graph.new_vp('double')
    delta = 1 + epsilon
    iteration = 1
    print(f'Beginning iteration with epsilon = {epsilon}')
    while delta >= epsilon:
        for i in tqdm(graph.vertices(), total=vertex_count, desc=f'Iteration {iteration}'):
            neighbor_diff_sum[i] = graph.get_all_neighbors(i, vprops=[diffs])[:, 1].sum()
        new_diffs = (neighbor_diff_sum.a + weight_term) / k
        # Unnecessary difficulty normalisation:
        # new_diffs /= new_diffs.sum()
        delta = np.linalg.norm(new_diffs - diffs.a, ord=1)
        diffs.a = new_diffs
        print(f'Completed with delta = {delta}')
        iteration += 1
    return diffs.a


def map_difficulties(file_location):
    graph = read_graph(file_location)
    names = vertex_names(graph)
    diffs = diffs_iterative(graph)
    return names, diffs


def normalize_difficulties(diffs):
    diffs /= np.mean(diffs)
    return diffs


def read_graph(file_location):
    graph = gt.load_graph(file_location, fmt='gt')
    print('Graph successfully read from file')
    return graph


def vertex_degrees(graph):
    degrees = graph.get_total_degrees(graph.get_vertices())
    print('Vertex degrees calculated from graph')
    return degrees


def vertex_names(graph):
    names = [tuple(graph.vp.name[v]) for v in graph.vertices()]
    print('Vertex names retrieved from graph')
    return names


if __name__ == '__main__':
    filename = 'comparison_graph.gpickle'
    node_list, difficulties = map_difficulties(filename)
    difficulties /= np.amax(difficulties)
