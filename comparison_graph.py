from itertools import combinations

import networkx as nx
import numpy as np

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
mod_comparison_threshold = 2

scores_db_loc, beatmaps_db_loc, beatmapsets_db_loc = db.db_location(game_mode, dump_type, dump_date)
playmode = funcs.play_mode(game_mode)
cursors = db.db_cursors(playmode, beatmaps_db_loc, beatmapsets_db_loc, scores_db_loc)
cur_beatmaps, cur_beatmapsets, cur_scores, cur_scores_single, cur_scores_acc_time = cursors
scores_table = db.db_tables(game_mode)[0]

beatmapsets = db.query_beatmapsets(cur_beatmapsets, ranked_status)
ranked_mods = tuple(funcs.allowed_mods(playmode))
for bms in beatmapsets:
    beatmaps = tuple(db.query_beatmaps(cur_beatmaps, bms))
    maps = db.query_maps(cur_scores, scores_table, beatmaps, ranked_mods)
    map_pairs = combinations(maps, 2)
    bms_graph = nx.DiGraph()
    # Even if a no-mod node will have order 0 now, it may form edges with no-mod nodes of other beatmapsets later.
    bms_graph.add_nodes_from({(b, 0) for b in beatmaps})
    for e in map_pairs:
        map_1, map_2 = e[0], e[1]
        threshold = mod_comparison_threshold if map_1[1] != 0 or map_2[1] != 0 else comparison_threshold
        shared_users = tuple(db.query_shared_users(cur_scores_single, scores_table, *map_1, *map_2))
        if len(shared_users) >= threshold:
            user_accs_1, user_times_1 = db.query_accs_times(cur_scores_acc_time, scores_table, shared_users, *map_1)
            user_accs_2, user_times_2 = db.query_accs_times(cur_scores_acc_time, scores_table, shared_users, *map_2)
            time_weights = funcs.timedelta_weights(user_times_1, user_times_2, weeks=8)
            t_stat = funcs.tstat_paired_weighted(user_accs_1, user_accs_2, time_weights)
            if not np.isnan(t_stat):
                bms_graph.add_edge(*e, weight=t_stat)
            else:
                print(f"Map pair {e} has undefined t-statistic (zero variance). Skipping edge formation.")
