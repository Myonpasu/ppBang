from itertools import combinations
from multiprocessing.dummy import Pool

import networkx as nx
from tqdm import tqdm

import database
from graph_functions import allowed_mods
from graph_functions import form_edge
from graph_functions import play_mode

# Game mode.
game_mode = 'standard'
# Beatmap difficulties will be calculated for the following beatmapset ranked statuses.
statuses = {'ranked', 'approved', 'loved'}
# Top 10000 players or random data dump?
dump_type = 'top'
# Date of dump to use.
dump_date = '2019_09_01'
# Number of common players required to make comparison. Must be strictly greater than 1.
threshold = 30
# Number of common players required to make comparison with game modifier. Must be strictly greater than 1.
mod_threshold = 3

# Preparation for database queries.
playmode = play_mode(game_mode)
beatmaps_db_loc, beatmapsets_db_loc, scores_db_loc = database.db_location(game_mode, dump_type, dump_date)
cur_beatmaps, cur_beatmapsets = database.db_cursors_beatmaps(beatmaps_db_loc, beatmapsets_db_loc)
cur_scores, cur_scores_single, cur_scores_acc_time = database.db_cursors_scores(playmode, scores_db_loc)
scores_table = database.db_tables(game_mode)[0]

# Initialise graph.
graph = nx.DiGraph()

# Form edges between all appropriate modified map pairs.
beatmapsets = list(database.query_beatmapsets(cur_beatmapsets, statuses))
ranked_mods = allowed_mods(playmode)
for bms in tqdm(beatmapsets, desc='Edges (with mods)'):
    beatmaps = tuple(database.query_beatmapset_beatmaps(cur_beatmaps, bms))
    maps = database.query_maps(cur_scores, scores_table, beatmaps, ranked_mods)
    for pair in combinations(maps, 2):
        if pair[0][1] != 0 and pair[1][1] != 0:
            form_edge(cur_scores_acc_time, cur_scores_single, scores_table, graph, *pair, mod_threshold)

# Form edges between all appropriate no-mod map pairs.
all_beatmaps = list(database.query_beatmaps(cur_beatmaps, cur_scores_single, scores_table, statuses, threshold, 0))
num_beatmaps = len(all_beatmaps)
num_beatmap_pairs = num_beatmaps * (num_beatmaps - 1) // 2
if __name__ == '__main__':
    with Pool(processes=4) as pool:
        for beatmap_pair in tqdm(combinations(all_beatmaps, 2), total=num_beatmap_pairs, desc='Edges (no mods)'):
            map_1, map_2 = (beatmap_pair[0], 0), (beatmap_pair[1], 0)
            cur_scores, cur_scores_single, cur_scores_acc_time = database.db_cursors_scores(playmode, scores_db_loc)
            form_edge_args = (cur_scores_acc_time, cur_scores_single, scores_table, graph, map_1, map_2, threshold)
            pool.apply_async(form_edge, form_edge_args)

# Write graph to file.
nx.write_graphml(graph, 'comparison_graph.graphml')
