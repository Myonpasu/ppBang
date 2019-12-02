from datetime import timedelta
from itertools import chain
from itertools import combinations
from itertools import product
from os import remove
from os.path import isfile

import networkx as nx
import numpy as np
from tqdm import tqdm

import database
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


def construct_graph(mode, dump_type, dump_date, statuses, threshold=30, mod_threshold=4):
    # Preparation for database queries.
    playmode = play_mode(mode)
    beatmaps_db_loc, beatmapsets_db_loc, scores_db_loc = database.db_location(mode, dump_type, dump_date)
    cur_beatmaps, cur_beatmapsets = database.db_cursors_beatmaps(beatmaps_db_loc, beatmapsets_db_loc)
    cur_scores, cur_scores_single, cur_scores_acc_time = database.db_cursors_scores(playmode, scores_db_loc)
    scores_table = database.db_tables(mode)[0]
    ranked_mods = allowed_mods(playmode)

    # Initialise graph.
    graph = nx.DiGraph()

    nomod_maps = database.query_maps_approved(cur_beatmaps, cur_scores_single, scores_table, statuses, threshold, 0)
    num_nomod_maps = len(nomod_maps)

    # Form edges between all appropriate mod map pairs.
    for mod in ranked_mods:
        if mod == 0:
            map_pairs = combinations(nomod_maps, 2)
            num_map_pairs = num_nomod_maps * (num_nomod_maps - 1) // 2
        else:
            query_maps_approved_args = cur_beatmaps, cur_scores_single, scores_table, statuses, mod_threshold, mod
            mod_maps = database.query_maps_approved(*query_maps_approved_args)
            num_mod_maps = len(mod_maps)
            mod_pairs = combinations(mod_maps, 2)
            mod_nomod_pairs = product(mod_maps, nomod_maps)
            map_pairs = chain(mod_pairs, mod_nomod_pairs)
            num_map_pairs = num_mod_maps * (num_nomod_maps + (num_mod_maps - 1) // 2)
        for pair in tqdm(map_pairs, total=num_map_pairs, desc=f'Edges ({readable_mod(mod)})'):
            map_1, map_2 = pair[0], pair[1]
            map_1_mod, map_2_mod = map_1[1], map_2[1]
            pair_threshold = mod_threshold if map_1_mod != 0 or map_2_mod != 0 else threshold
            cur_scores, cur_scores_single, cur_scores_acc_time = database.db_cursors_scores(playmode, scores_db_loc)
            form_edge_args = (cur_scores_acc_time, cur_scores_single, scores_table, graph, map_1, map_2, pair_threshold)
            form_edge(*form_edge_args)

    # Form edges between all appropriate intra-beatmapset map pairs.
    beatmapsets = list(database.query_beatmapsets(cur_beatmapsets, statuses))
    for bms in tqdm(beatmapsets, desc='Edges (intra-beatmapset)'):
        beatmaps = tuple(database.query_beatmapset_beatmaps(cur_beatmaps, bms))
        maps = database.query_maps(cur_scores, scores_table, beatmaps, ranked_mods)
        for pair in combinations(maps, 2):
            map_1_mod, map_2_mod = pair[0][1], pair[1][1]
            if map_1_mod != 0 and map_2_mod != 0 and map_1_mod != map_2_mod:
                form_edge(cur_scores_acc_time, cur_scores_single, scores_table, graph, *pair, mod_threshold)

    return graph


def form_edge(cur_scores_acc_time, cur_scores_single, scores_table, graph, map_1, map_2, threshold, report=False):
    """Form a directed edge between two maps if the number of common users exceeds a threshold."""
    shared_users = tuple(database.query_shared_users(cur_scores_single, scores_table, *map_1, *map_2))
    if len(shared_users) >= threshold:
        user_accs_1, user_times_1 = database.query_accs_times(cur_scores_acc_time, scores_table, shared_users, *map_1)
        user_accs_2, user_times_2 = database.query_accs_times(cur_scores_acc_time, scores_table, shared_users, *map_2)
        time_weights = timedelta_weights(user_times_1, user_times_2, weeks=8)
        if (time_weights >= 0.125).sum() >= threshold:  # Check at least threshold score pairs were within 24 weeks.
            t_stat = tstat_paired_weighted(user_accs_1, user_accs_2, time_weights)
            if not np.isnan(t_stat):
                graph.add_edge(map_1, map_2, weight=t_stat)
            elif report:
                print(f"Map pair has undefined t-statistic (zero variance). Skipping edge formation {(map_1, map_2)}.")


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


def timedelta_weights(user_times_1, user_times_2, weeks=8):
    weights = ((np.asarray(user_times_1) - np.asarray(user_times_2)) / timedelta(weeks=weeks)).astype('float')
    return np.exp2(- np.fabs(weights))


def tstat_paired_weighted(a, b, weights):
    """Calculates a t-statistic weighted by reliability weights.

    See https://stats.stackexchange.com/a/252167
    """
    data = np.asarray(a) - np.asarray(b)
    sum_weights = sum(weights)
    sum_weight_squares = sum(w * w for w in weights)
    mean = np.dot(data, weights) / sum_weights
    sumsquares = np.dot((data - mean) ** 2, weights)
    if sumsquares == 0:
        return np.nan
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
    data_dump_date = '2019_12_01'
    # Beatmap difficulties will be calculated for the following beatmapset ranked statuses.
    status_names = {'ranked', 'approved', 'loved'}
    # Output filename.
    filename = 'comparison_graph.gpickle'

    comparison_graph = construct_graph(game_mode, data_dump_type, data_dump_date, status_names)

    # Write graph to file.
    try:
        nx.write_gpickle(comparison_graph, filename)
        print(f'Graph successfully written as pickle to {filename}')
    except MemoryError:
        if isfile(filename):
            remove(filename)
        print('MemoryError occurred when writing pickle')
