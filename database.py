import sqlite3


def counts_to_accuracy(count50, count100, count300, countmiss, countgeki, countkatu, playmode=0):
    if playmode == 0:
        # osu!standard
        total_hits = count50 + count100 + count300 + countmiss
        acc = (count50 / 6 + count100 / 3 + count300)
        acc /= total_hits
    elif playmode == 1:
        # osu!taiko
        total_hits = count50 + count100 + count300 + countmiss
        acc = count100 / 2 + count300
        acc /= total_hits
    elif playmode == 2:
        # osu!catch
        total_hits = count50 + count100 + count300 + countmiss + countkatu
        if total_hits == 0:
            return 0
        total_successful_hits = count50 + count100 + count300
        acc = total_successful_hits / total_hits
    elif playmode == 3:
        # osu!mania
        total_hits = count50 + count100 + count300 + countmiss + countgeki + countkatu
        acc = count50 / 6 + count100 / 3 + countkatu * 2 / 3 + count300 + countgeki
        acc /= total_hits
    else:
        print("Invalid game mode.")
        return
    return acc


def db_cursors_multi(playmode, beatmaps_db_location, scores_db_location):
    con_beatmaps = sqlite3.connect(beatmaps_db_location)
    con_beatmaps.execute("PRAGMA mmap_size=67108864")
    con_beatmaps.row_factory = lambda cursor, row: row[0]
    cur_beatmaps = con_beatmaps.cursor()
    con_scores = sqlite3.connect(scores_db_location, detect_types=sqlite3.PARSE_DECLTYPES)
    con_scores.execute("PRAGMA mmap_size=8589934592")
    con_scores.row_factory = lambda cursor, row: row[0]
    cur_scores_single = con_scores.cursor()
    con_scores.row_factory = lambda cursor, row: (counts_to_accuracy(*row[0:6], playmode=playmode), row[6])
    cur_scores_acc_time = con_scores.cursor()
    return cur_beatmaps, cur_scores_single, cur_scores_acc_time


def db_cursors_single(beatmapsets_db_location, scores_db_location):
    con_beatmapsets = sqlite3.connect(beatmapsets_db_location)
    con_beatmapsets.row_factory = lambda cursor, row: row[0]
    cur_beatmapsets = con_beatmapsets.cursor()
    con_scores = sqlite3.connect(scores_db_location)
    con_scores.execute("PRAGMA mmap_size=8589934592")
    cur_scores = con_scores.cursor()
    return cur_beatmapsets, cur_scores


def db_location(game_mode, dump_type, dump_date):
    if game_mode == 'standard':
        mode_str = '_'
        mode_folder_str = '_osu_'
    elif game_mode in ['taiko', 'fruits', 'mania']:
        mode_folder_str = mode_str = f'_{game_mode}_'
    else:
        print('Invalid game mode specified.')
        return
    beatmaps_db_location = f"./{dump_date}_performance{mode_folder_str}{dump_type}/osu_beatmaps.db"
    beatmapsets_db_location = f"./{dump_date}_performance{mode_folder_str}{dump_type}/osu_beatmapsets.db"
    scores_db_location = f"./{dump_date}_performance{mode_folder_str}{dump_type}/osu_scores{mode_str}high.db"
    return beatmaps_db_location, beatmapsets_db_location, scores_db_location


def db_tables(game_mode):
    scores_table = 'osu_scores'
    user_stats_table = 'osu_user_stats'
    if game_mode != 'standard':
        game_mode_str = f'_{game_mode}'
        scores_table += game_mode_str
        user_stats_table += game_mode_str
    scores_table += '_high'
    return scores_table, user_stats_table


def query_accs_times(cursor, scores_table, users, beatmap_id, enabled_mods):
    query = f"SELECT count50, count100, count300, countmiss, countgeki, countkatu, date FROM {scores_table} " \
            f"WHERE user_id IN {users} AND beatmap_id == ? AND enabled_mods == ?"
    user_scores = list(cursor.execute(query, (beatmap_id, enabled_mods)))
    user_accs, user_times = zip(*user_scores)
    return user_accs, user_times


def query_beatmapset_beatmaps(cursor, beatmapset):
    beatmaps = cursor.execute("SELECT beatmap_id FROM osu_beatmaps WHERE beatmapset_id == ?", (beatmapset,))
    return beatmaps


def query_beatmapsets(cursor, status_names):
    statuses = tuple(ranked_status(status_names))
    beatmapsets = cursor.execute(
        f"SELECT beatmapset_id FROM osu_beatmapsets WHERE approved IN ({', '.join('?' * len(statuses))})", statuses)
    return beatmapsets


def query_maps(cursor, scores_table, beatmaps, mods):
    query = f"SELECT DISTINCT beatmap_id, enabled_mods FROM {scores_table} " \
            f"WHERE beatmap_id IN ({', '.join('?' * len(beatmaps))}) AND enabled_mods IN ({', '.join('?' * len(mods))})"
    maps = cursor.execute(query, beatmaps + mods)
    return maps


def query_maps_approved(cur_beatmaps, cur_scores_single, scores_table, status_names, threshold, enabled_mod):
    filtered_beatmaps = tuple(cur_scores_single.execute(
        f"SELECT beatmap_id FROM {scores_table} "
        "GROUP BY beatmap_id "
        "HAVING SUM(CASE enabled_mods WHEN ? THEN 1 ELSE 0 END) >= ?", (enabled_mod, threshold)))
    # Correctly format query when searching for only a single beatmap.
    if len(filtered_beatmaps) == 1:
        filtered_beatmaps = f'({filtered_beatmaps[0]})'
    statuses = tuple(ranked_status(status_names))
    beatmaps = cur_beatmaps.execute(
        "SELECT beatmap_id FROM osu_beatmaps "
        f"WHERE approved IN ({', '.join('?' * len(statuses))}) "
        f"AND beatmap_id IN {filtered_beatmaps}", statuses)
    maps = [(beatmap_id, enabled_mod) for beatmap_id in beatmaps]
    return maps


def query_shared_users(cursor, scores_table, beatmap_id_1, enabled_mods_1, beatmap_id_2, enabled_mods_2):
    query = f"SELECT user_id FROM {scores_table} " \
            "WHERE beatmap_id == ? AND enabled_mods == ? " \
            "INTERSECT " \
            f"SELECT user_id FROM {scores_table} " \
            "WHERE beatmap_id == ? AND enabled_mods == ?"
    users = cursor.execute(query, (beatmap_id_1, enabled_mods_1, beatmap_id_2, enabled_mods_2))
    return users


def ranked_status(status_names):
    statuses = set()
    for s in status_names:
        if s == 'ranked':
            statuses.add(1)
        elif s == 'approved':
            statuses.add(2)
        elif s == 'qualified':
            statuses.add(3)
        elif s == 'loved':
            statuses.add(4)
    return statuses
