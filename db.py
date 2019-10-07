import sqlite3

from funcs import counts_to_accuracy
from funcs import ranked_status


def db_cursors(playmode, beatmaps_db_location, beatmapsets_db_location, scores_db_location):
    con_beatmaps = sqlite3.connect(beatmaps_db_location)
    con_beatmaps.row_factory = lambda cursor, row: row[0]
    cur_beatmaps = con_beatmaps.cursor()
    con_beatmapsets = sqlite3.connect(beatmapsets_db_location)
    con_beatmapsets.row_factory = lambda cursor, row: row[0]
    cur_beatmapsets = con_beatmapsets.cursor()
    con_scores = sqlite3.connect(scores_db_location, detect_types=sqlite3.PARSE_DECLTYPES)
    con_scores.execute("PRAGMA mmap_size=8589934592")
    con_scores.execute("PRAGMA locking_mode=EXCLUSIVE")
    cur_scores = con_scores.cursor()
    con_scores.row_factory = lambda cursor, row: row[0]
    cur_scores_single = con_scores.cursor()
    con_scores.row_factory = lambda cursor, row: (counts_to_accuracy(*row[0:6], playmode=playmode), row[6])
    cur_scores_acc_time = con_scores.cursor()
    return cur_beatmaps, cur_beatmapsets, cur_scores, cur_scores_single, cur_scores_acc_time


def db_location(game_mode, dump_type, dump_date):
    if game_mode == 'standard':
        mode_str = '_'
        mode_folder_str = '_osu_'
    elif game_mode in ['taiko', 'fruits', 'mania']:
        mode_folder_str = mode_str = f'_{game_mode}_'
    else:
        print('Invalid game mode specified.')
        return
    scores_db_location = f"./{dump_date}_performance{mode_folder_str}{dump_type}/osu_scores{mode_str}high.db"
    beatmaps_db_location = f"./{dump_date}_performance{mode_folder_str}{dump_type}/osu_beatmaps.db"
    beatmapsets_db_location = f"./{dump_date}_performance{mode_folder_str}{dump_type}/osu_beatmapsets.db"
    return scores_db_location, beatmaps_db_location, beatmapsets_db_location


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
    user_scores = list(cursor.execute(
        "SELECT count50, count100, count300, countmiss, countgeki, countkatu, date FROM ? "
        f"WHERE user_id IN {users} AND beatmap_id == ? AND enabled_mods == ?",
        (scores_table, beatmap_id, enabled_mods)))
    user_accs, user_times = zip(*user_scores)
    return user_accs, user_times


def query_beatmaps(cursor, beatmapset):
    beatmaps = cursor.execute("SELECT beatmap_id FROM osu_beatmaps WHERE beatmapset_id == ?", (beatmapset,))
    return beatmaps


def query_beatmapsets(cursor, status):
    approved = tuple(ranked_status(status))
    beatmapsets = cursor.execute(
        f"SELECT beatmapset_id FROM osu_beatmapsets WHERE approved IN ({', '.join('?' * len(status))})", approved)
    return beatmapsets


def query_maps(cursor, scores_table, beatmaps, mods):
    maps = cursor.execute(
        f"SELECT DISTINCT beatmap_id, enabled_mods FROM ? WHERE beatmap_id IN {beatmaps} AND enabled_mods IN {mods}",
        (scores_table,))
    return maps


def query_shared_users(cursor, scores_table, beatmap_id_1, enabled_mods_1, beatmap_id_2, enabled_mods_2):
    users = cursor.execute(
        "SELECT user_id FROM ? "
        "WHERE beatmap_id == ? AND enabled_mods == ? "
        "INTERSECT "
        "SELECT user_id FROM ? "
        "WHERE beatmap_id == ? AND enabled_mods == ?",
        (scores_table, beatmap_id_1, enabled_mods_1, scores_table, beatmap_id_2, enabled_mods_2))
    return users
