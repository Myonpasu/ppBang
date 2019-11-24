from itertools import chain
from itertools import combinations
from itertools import product
from os import remove
from os.path import isfile

import networkx as nx
from tqdm import tqdm

import database
from graph_functions import allowed_mods
from graph_functions import form_edge
from graph_functions import play_mode
from output_functions import readable_mod


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


if __name__ == '__main__':
    # Game mode.
    game_mode = 'standard'
    # Top 10000 players ('top') or random ('random') data dump?
    data_dump_type = 'top'
    # Date of dump to use.
    data_dump_date = '2019_09_01'
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
