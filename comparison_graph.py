from itertools import combinations

import networkx as nx
from tqdm import tqdm

import database
import graph_functions

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
playmode = graph_functions.play_mode(game_mode)
beatmaps_db_loc, beatmapsets_db_loc, scores_db_loc = database.db_location(game_mode, dump_type, dump_date)
cursors = database.db_cursors(playmode, beatmaps_db_loc, beatmapsets_db_loc, scores_db_loc)
cur_beatmaps, cur_beatmapsets, cur_scores, cur_scores_single, cur_scores_acc_time = cursors
scores_table = database.db_tables(game_mode)[0]

# Initialise graph.
graph = nx.DiGraph()

# Form edges between all appropriate modified map pairs.
beatmapsets = list(database.query_beatmapsets(cur_beatmapsets, statuses))
ranked_mods = graph_functions.allowed_mods(playmode)
for bms in tqdm(beatmapsets, desc='Edges (with mods)'):
    beatmaps = tuple(database.query_beatmapset_beatmaps(cur_beatmaps, bms))
    maps = database.query_maps(cur_scores, scores_table, beatmaps, ranked_mods)
    for pair in combinations(maps, 2):
        if pair[0][1] != 0 and pair[1][1] != 0:
            graph_functions.form_edge(cur_scores_acc_time, cur_scores_single, scores_table, graph, *pair, mod_threshold)

# Form edges between all appropriate no-mod map pairs.
all_beatmaps = list(database.query_beatmaps(cur_beatmaps, cur_scores_single, scores_table, statuses, threshold, 0))
num_beatmaps = len(all_beatmaps)
num_beatmap_pairs = num_beatmaps * (num_beatmaps - 1) // 2
for beatmap_pair in tqdm(combinations(all_beatmaps, 2), total=num_beatmap_pairs, desc='Edges (no mods)'):
    map_1, map_2 = (beatmap_pair[0], 0), (beatmap_pair[1], 0)
    graph_functions.form_edge(cur_scores_acc_time, cur_scores_single, scores_table, graph, map_1, map_2, threshold)

# Write graph to file.
nx.write_graphml(graph, 'comparison_graph.graphml')
