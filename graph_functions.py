import itertools
from datetime import timedelta

import numpy as np

import database


def allowed_mods(playmode):
    """Return the set of possible enabled mods for a game mode."""
    # None = 0, NF = 1, EZ = 2, HD = 8, HR = 16, DT = 64, HT = 256, NC = 512, FL = 1024, FI = 1048576.
    # NC is only set along with DT, giving 576.
    mods = [2, 8, 16, 64, 256, 1024, 1048576] if playmode == 3 else [2, 8, 16, 64, 256, 1024]
    mod_powerset = itertools.chain.from_iterable(itertools.combinations(mods, r) for r in range(len(mods) + 1))
    if playmode == 3:
        combos = (p for p in mod_powerset if
                  not ((2 in p and 16 in p) or (64 in p and 256 in p) or (8 in p and 1048576 in p)))
    else:
        combos = (p for p in mod_powerset if not ((2 in p and 16 in p) or (64 in p and 256 in p)))
    allowed = tuple(sum(c) for c in combos)
    return allowed


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


def form_edge(cur_scores_acc_time, cur_scores_single, scores_table, graph, map_1, map_2, threshold, report=False):
    """Form a directed edge between two maps if the number of common users exceeds a threshold."""
    shared_users = tuple(database.query_shared_users(cur_scores_single, scores_table, *map_1, *map_2))
    if len(shared_users) >= threshold:
        user_accs_1, user_times_1 = database.query_accs_times(cur_scores_acc_time, scores_table, shared_users, *map_1)
        user_accs_2, user_times_2 = database.query_accs_times(cur_scores_acc_time, scores_table, shared_users, *map_2)
        time_weights = timedelta_weights(user_times_1, user_times_2, weeks=8)
        t_stat = tstat_paired_weighted(user_accs_1, user_accs_2, time_weights)
        if not np.isnan(t_stat):
            graph.add_edge(map_1, map_2, weight=t_stat)
        elif report:
            print(f"Map pair has undefined t-statistic (zero variance). Skipping edge formation {(map_1, map_2)}.")


def play_mode(game_mode):
    if game_mode == 'standard':
        playmode = 0
    elif game_mode == 'taiko':
        playmode = 1
    elif game_mode == 'fruits':
        playmode = 2
    elif game_mode == 'mania':
        playmode = 3
    else:
        print("Invalid game mode.")
        return
    return playmode


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
