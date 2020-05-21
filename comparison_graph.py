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
from output_functions import readable_mod


def allowed_mods(playmode):
    """Return the set of possible enabled mods for a game mode, including no-mod."""
    # None = 0, NF = 1, EZ = 2, HD = 8, HR = 16, DT = 64, HT = 256, NC = 512, FL = 1024, FI = 1048576.
    # NC is only set along with DT, giving 576.
    mods = [2, 8, 16, 64, 256, 1024, 1048576] if playmode == 3 else [2, 8, 16, 64, 256, 1024]
    mod_powerset = chain.from_iterable(combinations(mods, r) for r in range(len(mods) + 1))
    if playmode == 3:
        combos = (p for p in mod_powerset if
                  not ((2 in p and 16 in p) or (64 in p and 256 in p) or (8 in p and 1048576 in p)))
    else:
        combos = (p for p in mod_powerset if not ((2 in p and 16 in p) or (64 in p and 256 in p)))
    allowed = tuple(sum(c) for c in combos)
    return allowed


def construct_graph(mode, dump_type, dump_date, statuses, threshold=30, mod_threshold=5):
    # Preparation for database queries.
    playmode = play_mode(mode)
    beatmaps_db_loc, beatmapsets_db_loc, scores_db_loc, attribs_db_loc = db_location(mode, dump_type, dump_date)
    cur_beatmaps, cur_scores_single, cur_scores_acc_time = db_cursors_multi(playmode, beatmaps_db_loc, scores_db_loc)
    cur_beatmapsets, cur_scores = db_cursors_single(beatmapsets_db_loc, scores_db_loc)
    scores_table = db_tables(mode)[0]
    ranked_mods = allowed_mods(playmode)

    # Form edges between all appropriate mod map pairs.
    nomod_maps = query_maps_approved(cur_beatmaps, cur_scores_single, scores_table, statuses, threshold, 0)
    num_nomod_maps = len(nomod_maps)
    mod_graph_args = ((m, nomod_maps, num_nomod_maps, beatmaps_db_loc, scores_db_loc, scores_table,
                       playmode, statuses, threshold, mod_threshold) for m in ranked_mods)
    with ProcessPoolExecutor() as executor:
        mod_graphs = list(executor.map(mod_graph_wrapper, mod_graph_args))

    # Form edges between all appropriate intra-beatmapset map pairs.
    beatmapsets = list(query_beatmapsets(cur_beatmapsets, statuses))
    intrabms_graph_args = (beatmapsets, ranked_mods, mod_threshold, cur_beatmaps, cur_scores,
                           cur_scores_acc_time, cur_scores_single, scores_table)
    intrabms_graph = graph_intrabms(*intrabms_graph_args)
    intrabms_filename = f'temp_{playmode}_INTRABMS.gt'
    intrabms_graph.save(intrabms_filename, fmt='gt')

    # Combine all graphs.
    mod_graphs.append(intrabms_graph)
    graph = graph_union_all(mod_graphs)
    return graph


def form_edge(cur_scores_acc_time, cur_scores_single, scores_tab, graph, map_1, map_2, i_1, i_2, thresh, report=False):
    """Form a directed edge between two maps if the number of common users exceeds a threshold."""
    shared_users = tuple(query_shared_users(cur_scores_single, scores_tab, *map_1, *map_2))
    if len(shared_users) >= thresh:
        user_accs_1, user_times_1 = query_accs_times(cur_scores_acc_time, scores_tab, shared_users, *map_1)
        user_accs_2, user_times_2 = query_accs_times(cur_scores_acc_time, scores_tab, shared_users, *map_2)
        time_weights = timedelta_weights(user_times_1, user_times_2, weeks=8)
        if (time_weights >= 0.125).sum() >= thresh:  # Check at least threshold score pairs were within 24 weeks.
            t_stat = tstat_paired_weighted(user_accs_1, user_accs_2, time_weights)
            if not np.isnan(t_stat):
                e = graph.add_edge(i_1, i_2)
                graph.ep.weight[e] = t_stat
            elif report:
                print(f"Map pair has undefined t-statistic (zero variance). Skipping edge formation {(map_1, map_2)}.")


def graph_add_names(graph, vertex_names):
    isolated_vertices = [v for v in graph.vertices() if v.in_degree() + v.out_degree() == 0]
    for v in graph.vertices():  # Assign name to every non-isolated vertex.
        if v not in isolated_vertices:
            graph.vp.name[v] = vertex_names[graph.vertex_index[v]]
    graph.remove_vertex(isolated_vertices, fast=True)


def graph_intrabms(beatmapsets, ranked_mods, mod_thres, cur_beatmaps, cur_scores, cur_acc_time, cur_single, scores_tab):
    intrabms_graph = graph_spawn()
    for bms in tqdm(beatmapsets, desc='Edges (intra-beatmapset)'):
        bms_graph = graph_spawn()
        beatmaps = tuple(query_beatmapset_beatmaps(cur_beatmaps, bms))
        maps = list(query_maps(cur_scores, scores_tab, beatmaps, ranked_mods))
        for idx_1, idx_2 in combinations(range(len(maps)), 2):
            map_1, map_2 = maps[idx_1], maps[idx_2]
            map_1_mod, map_2_mod = map_1[1], map_2[1]
            if map_1_mod != 0 and map_2_mod != 0 and map_1_mod != map_2_mod:
                form_edge(cur_acc_time, cur_single, scores_tab, bms_graph,
                          map_1, map_2, idx_1, idx_2, mod_thres)
        graph_add_names(bms_graph, maps)
        intrabms_graph = graph_union_pair(intrabms_graph, bms_graph)
    return intrabms_graph


def graph_spawn():
    """Return an empty directed graph with internal name and weight properties."""
    graph = gt.Graph(directed=True)
    graph.vp.name = graph.new_vp('vector<int>')
    graph.ep.weight = graph.new_ep('double')
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


def mod_graph(mod, nomod_maps, num_nomod_maps, beatmaps_db, scores_db, scores_tab, mode, statuses, thresh, mod_thresh):
    graph = graph_spawn()
    cur_beatmaps, cur_scores_single, cur_scores_acc_time = db_cursors_multi(mode, beatmaps_db, scores_db)
    nomod_indices = range(num_nomod_maps)
    if mod == 0:
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
        pair_threshold = mod_thresh if map_1_mod != 0 or map_2_mod != 0 else thresh
        form_edge(cur_scores_acc_time, cur_scores_single, scores_tab, graph, map_1, map_2, idx_1, idx_2, pair_threshold)
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


def timedelta_weights(user_times_1, user_times_2, weeks=8, cutoff_weeks=24):
    time_delta = ((np.asarray(user_times_1) - np.asarray(user_times_2)) / timedelta(weeks=weeks)).astype('float')
    weights = np.exp2(- np.fabs(time_delta))
    weights[weights <= np.exp2(- cutoff_weeks / weeks)] = 0  # Set weights for large enough time intervals to zero.
    return weights


def tstat_paired_weighted(a, b, weights):
    """Calculates a t-statistic weighted by reliability weights.

    See https://stats.stackexchange.com/a/252167
    """
    data = np.asarray(a) - np.asarray(b)
    sum_weights = sum(weights)
    sum_weight_squares = sum(w * w for w in weights)
    mean = np.dot(data, weights) / sum_weights
    if np.all(np.isclose(np.ma.masked_where(weights == 0, data), mean)):
        return np.nan
    sumsquares = np.dot((data - mean) ** 2, weights)
    correction = sum_weights - sum_weight_squares / sum_weights
    if correction == 0:
        return np.nan
    var = sumsquares / correction
    std = np.sqrt(var)
    std_mean = std * np.sqrt(sum_weight_squares) / sum_weights
    tstat = mean / std_mean
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
