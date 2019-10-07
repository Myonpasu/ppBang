from itertools import combinations
from itertools import product

import networkx as nx

import db
import funcs

# Game mode.
game_mode = 'standard'
# Beatmap difficulties will be calculated for the following beatmapset ranked statuses.
ranked_status = {'ranked', 'approved', 'loved'}
# Top 10000 players or random data dump?
dump_type = 'top'
# Date of dump to use.
dump_date = '2019_09_01'
# Number of common players required to make comparison. Must be strictly greater than 1.
comparison_threshold = 30
# Number of common players required to make comparison with game modifier. Must be strictly greater than 1.
mod_comparison_threshold = 3

# Preparation for database queries.
playmode = funcs.play_mode(game_mode)
beatmaps_db_loc, beatmapsets_db_loc, scores_db_loc = db.db_location(game_mode, dump_type, dump_date)
cursors = db.db_cursors(playmode, beatmaps_db_loc, beatmapsets_db_loc, scores_db_loc)
cur_beatmaps, cur_beatmapsets, cur_scores, cur_scores_single, cur_scores_acc_time = cursors
scores_table = db.db_tables(game_mode)[0]

beatmapsets = list(db.query_beatmapsets(cur_beatmapsets, ranked_status))
num_beatmapsets = len(beatmapsets)
processed = 0
ranked_mods = tuple(funcs.allowed_mods(playmode))
graph = nx.DiGraph()
existing_beatmaps = ()
for bms in beatmapsets:
    beatmaps = tuple(db.query_beatmaps(cur_beatmaps, bms))
    maps = db.query_maps(cur_scores, scores_table, beatmaps, ranked_mods)
    bms_graph = nx.DiGraph()
    for e in combinations(maps, 2):
        map_1, map_2 = e[0], e[1]
        threshold = mod_comparison_threshold if map_1[1] != 0 or map_2[1] != 0 else comparison_threshold
        funcs.form_edge(cur_scores_acc_time, cur_scores_single, scores_table, bms_graph, map_1, map_2, threshold)
    graph.update(bms_graph)
    for p in product(beatmaps, existing_beatmaps):
        map_1, map_2 = (p[0], 0), (p[1], 0)
        funcs.form_edge(cur_scores_acc_time, cur_scores_single, scores_table, graph, map_1, map_2, comparison_threshold)
    existing_beatmaps += beatmaps
    processed += 1
    print(f"{processed / num_beatmapsets : .3%} ({processed} / {num_beatmapsets}) beatmapsets processed.")
