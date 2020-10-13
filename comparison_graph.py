from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from itertools import chain
from itertools import combinations
from itertools import product
from os import remove
from os.path import isfile

import graph_tool as gt
import numpy as np
from graph_tool.generation import graph_union
from tqdm import tqdm

from database import *
from mods import Mod
from mods import allowed_mods
from mods import readable_mod


def construct_graph(mode, dump_type, dump_date, statuses, threshold=30, mod_threshold=6):
    # Preparation for database queries.
    playmode = play_mode(mode)
    beatmaps_db_loc, beatmapsets_loc, scores_loc, attribs_loc = db_location(mode, dump_type, dump_date)
    cur_beatmaps, cur_scores_single, cur_scores_data = db_cursors_multi(playmode, beatmaps_db_loc, scores_loc)
    cur_beatmapsets, cur_attribs, cur_scores = db_cursors_single(beatmapsets_loc, attribs_loc, scores_loc)
    scores_table = db_tables(mode)[0]
    ranked_mods = allowed_mods()
    fc_dict = dict(full_combos(cur_attribs))

    # Form edges between all appropriate mod map pairs.
    nomod_maps = query_maps_approved(cur_beatmaps, cur_scores_single, scores_table, statuses, threshold, Mod(0))
    num_nomod_maps = len(nomod_maps)
    mod_graph_args = ((m, nomod_maps, num_nomod_maps, beatmaps_db_loc, scores_loc, scores_table,
                       playmode, statuses, fc_dict, threshold, mod_threshold) for m in ranked_mods)
    with ProcessPoolExecutor() as executor:
        mod_graphs = list(executor.map(mod_graph_wrapper, mod_graph_args))

    # Form edges between all appropriate intra-beatmapset map pairs.
    beatmapsets = list(query_beatmapsets(cur_beatmapsets, statuses))
    intrabms_graph_args = (beatmapsets, ranked_mods, fc_dict, mod_threshold, cur_beatmaps, cur_scores,
                           cur_scores_data, cur_scores_single, scores_table)
    intrabms_graph = graph_intrabms(*intrabms_graph_args)
    intrabms_filename = f'temp_{playmode}_INTRABMS.gt'
    intrabms_graph.save(intrabms_filename, fmt='gt')

    # Combine all graphs.
    mod_graphs.append(intrabms_graph)
    mod_graphs[:] = [g for g in mod_graphs if g.num_vertices() != 0]
    graph = graph_union_all(mod_graphs)
    return graph


def form_edge(cur_scores_data, cur_scores_single, scores_tab, graph, map_1, map_2, i1, i2, fcs, thresh, report=False):
    """Form a directed edge between two maps if the number of common users exceeds a threshold."""
    shared_users = tuple(query_shared_users(cur_scores_single, scores_tab, *map_1, *map_2))
    if len(shared_users) >= thresh:
        accs_1, hit_props_1, hit_accs_1, coms_1, times_1 = query_data(cur_scores_data, scores_tab, shared_users, *map_1)
        accs_2, hit_props_2, hit_accs_2, coms_2, times_2 = query_data(cur_scores_data, scores_tab, shared_users, *map_2)
        time_weights = timedelta_weights(times_1, times_2)
        if (time_weights > 0).sum() >= thresh:  # Check at least threshold score pairs were set within cutoff time.
            combos_1 = np.asarray(coms_1) / fcs[map_1[0]]
            combos_2 = np.asarray(coms_2) / fcs[map_2[0]]
            t_acc = tstat_paired_weighted(accs_1, accs_2, time_weights)
            t_hit_props = tstat_paired_weighted(hit_props_1, hit_props_2, time_weights)
            t_hit_accs = tstat_paired_weighted(hit_accs_1, hit_accs_2, time_weights)
            t_combo = tstat_paired_weighted(combos_1, combos_2, time_weights)
            weights = {
                'weight_accuracy': t_acc,
                'weight_hit_proportion': t_hit_props,
                'weight_hit_accuracy': t_hit_accs,
                'weight_combo': t_combo
            }
            if not np.any(np.isnan(list(weights.values()))):
                v1 = graph.vertex(i1, add_missing=True)
                v2 = graph.vertex(i2, add_missing=True)
                for prop, w in weights.items():
                    e = graph.edge(v1, v2, add_missing=True)
                    graph.ep[prop][e] = w
            elif report:
                print(f"Map pair has an undefined t-statistic. Skipping edge formation {(map_1, map_2)}.")


def graph_add_names(graph, vertex_names):
    isolated_vertices = [v for v in graph.vertices() if v.in_degree() + v.out_degree() == 0]
    for v in graph.vertices():  # Assign name to every non-isolated vertex.
        if v not in isolated_vertices:
            graph.vp.name[v] = vertex_names[graph.vertex_index[v]]
    graph.remove_vertex(isolated_vertices, fast=True)


def graph_intrabms(bmsets, ranked_mods, fcs, mod_thres, cur_beatmaps, cur_scores, cur_data, cur_single, scores_tab):
    intrabms_graph = graph_spawn()
    for bms in tqdm(bmsets, desc='Edges (intra-beatmapset)'):
        bms_graph = graph_spawn()
        beatmaps = tuple(query_beatmapset_beatmaps(cur_beatmaps, bms))
        maps = list(query_maps(cur_scores, scores_tab, beatmaps, ranked_mods))
        for idx_1, idx_2 in combinations(range(len(maps)), 2):
            map_1, map_2 = maps[idx_1], maps[idx_2]
            map_1_mod, map_2_mod = map_1[1], map_2[1]
            if map_1_mod != 0 and map_2_mod != 0 and map_1_mod != map_2_mod:
                form_edge(cur_data, cur_single, scores_tab, bms_graph,
                          map_1, map_2, idx_1, idx_2, fcs, mod_thres)
        graph_add_names(bms_graph, maps)
        intrabms_graph = graph_union_pair(intrabms_graph, bms_graph)
    return intrabms_graph


def graph_spawn():
    """Return an empty directed graph with internal name and weight properties."""
    graph = gt.Graph(directed=True)
    graph.vp.name = graph.new_vp('vector<int>')
    graph.ep.weight_accuracy = graph.new_ep('double', val=0)
    graph.ep.weight_hit_proportion = graph.new_ep('double', val=0)
    graph.ep.weight_hit_accuracy = graph.new_ep('double', val=0)
    graph.ep.weight_combo = graph.new_ep('double', val=0)
    return graph


def graph_union_all(graphs):
    print('Performing union over all graphs...')
    num_graphs = len(graphs)
    graphs = iter(graphs)
    u = next(graphs)  # Better to deep copy here, but it is too resource heavy.
    for g in tqdm(graphs, desc='Graph union', total=num_graphs - 1):
        u = graph_union_pair(u, g)
    return u


def graph_union_pair(g, h):
    """Union of two graphs with named vertices.

    Inserts h into g, which is modified, and returns g.
    """
    g_name_index = {g.vp.name[v]: g.vertex_index[v] for v in g.vertices()}
    intersect_vals = [g_name_index.get(h.vp.name[v], -1) for v in h.vertices()]
    intersect_prop = h.new_vp('unsigned long', vals=intersect_vals)
    u = graph_union(g, h, intersection=intersect_prop, include=True, internal_props=True)
    return u


def mod_graph(mod, nomod_maps, num_nomod_maps, bmap_db, scores_db, scores_tab, mode, statuses, fcs, thresh, mod_thresh):
    graph = graph_spawn()
    cur_beatmaps, cur_scores_single, cur_scores_data = db_cursors_multi(mode, bmap_db, scores_db)
    nomod_indices = range(num_nomod_maps)
    if not mod:  # No mods are enabled.
        maps = nomod_maps
        index_pairs = combinations(nomod_indices, 2)
        num_map_pairs = num_nomod_maps * (num_nomod_maps - 1) // 2
    else:
        query_maps_approved_args = cur_beatmaps, cur_scores_single, scores_tab, statuses, mod_thresh, mod
        mod_maps = query_maps_approved(*query_maps_approved_args)
        maps = nomod_maps + mod_maps
        num_mod_maps = len(mod_maps)
        num_maps = num_nomod_maps + num_mod_maps
        mod_indices = range(num_nomod_maps, num_maps)
        mod_pairs = combinations(mod_indices, 2)
        mod_nomod_pairs = product(mod_indices, nomod_indices)
        index_pairs = chain(mod_pairs, mod_nomod_pairs)
        num_map_pairs = num_mod_maps * num_nomod_maps + num_mod_maps * (num_mod_maps - 1) // 2
    mod_name = readable_mod(mod)
    for idx_1, idx_2 in tqdm(index_pairs, desc=f'Edges ({mod_name})', total=num_map_pairs):  # Form appropriate edges.
        map_1, map_2 = maps[idx_1], maps[idx_2]
        map_1_mod, map_2_mod = map_1[1], map_2[1]
        pair_thresh = mod_thresh if map_1_mod != 0 or map_2_mod != 0 else thresh
        form_edge(cur_scores_data, cur_scores_single, scores_tab, graph, map_1, map_2, idx_1, idx_2, fcs, pair_thresh)
    graph_add_names(graph, maps)
    filename = f'temp_{mode}_{mod_name}.gt'
    graph.save(filename, fmt='gt')
    return graph


def mod_graph_wrapper(arguments):
    return mod_graph(*arguments)


def play_mode(mode):
    if mode == 'standard':
        playmode = 0
    elif mode == 'taiko':
        playmode = 1
    elif mode == 'fruits':
        playmode = 2
    elif mode == 'mania':
        playmode = 3
    else:
        print("Invalid game mode.")
        return
    return playmode


def timedelta_weights(user_times_1, user_times_2, cutoff_weeks=36):
    time_delta = ((np.asarray(user_times_1) - np.asarray(user_times_2)) / timedelta(weeks=cutoff_weeks)).astype('float')
    time_delta = np.fabs(time_delta)
    weights = np.zeros(len(time_delta))
    cutoff_mask = time_delta < 1
    weights[cutoff_mask] = np.exp(2 / (1 - 1 / time_delta[cutoff_mask]))
    return weights


def tstat_paired_weighted(a, b, weights):
    """Calculates a t-statistic weighted by reliability weights.

    See https://stats.stackexchange.com/a/252167
    """
    data = np.asarray(a) - np.asarray(b)
    sum_weights = sum(weights)
    sum_weight_squares = sum(w * w for w in weights)
    mean = np.dot(data, weights) / sum_weights
    if np.allclose(data[weights != 0], mean):
        return np.nan
    sumsquares = np.dot((data - mean) ** 2, weights)
    correction = sum_weights - sum_weight_squares / sum_weights  # Never zero provided at least two nonzero weights.
    var = sumsquares / correction
    # Remove cases where the corrected sample standard deviation is at most 1% (an arbitrary choice).
    # Perhaps we should find a more natural way to rule out these divergent cases.
    if np.isclose(var, 0, atol=1e-4):
        return np.nan
    std = np.sqrt(var)
    std_err_mean = std * np.sqrt(sum_weight_squares) / sum_weights
    tstat = mean / std_err_mean
    return tstat


if __name__ == '__main__':
    # Game mode.
    game_mode = 'standard'
    # Top 10000 players ('top') or random ('random') data dump?
    data_dump_type = 'top'
    # Date of dump to use.
    data_dump_date = '2020_03_01'
    # Beatmap difficulties will be calculated for the following beatmapset ranked statuses.
    status_names = {'ranked', 'approved', 'loved'}
    # Output filename and extension.
    file_name = 'comparison_graph'
    file_extension = 'gt'

    # Construct comparison graph in memory.
    comparison_graph = construct_graph(game_mode, data_dump_type, data_dump_date, status_names)
    print('Comparison graph constructed')

    # Format file name depending on game mode and data dump date.
    if game_mode != 'standard':
        file_name += f'_{game_mode}'
    file_name += f'_{data_dump_date}.{file_extension}'
    # Write graph to file.
    try:
        comparison_graph.save(file_name, fmt='gt')
        print(f"Graph successfully saved to '{file_name}'")
    except MemoryError:
        if isfile(file_name):
            remove(file_name)
        print('MemoryError occurred while saving graph')
