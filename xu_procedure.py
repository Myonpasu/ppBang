import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix


def sparse_solve_cg(a, b, precondition=False):
    """Solve linear equations a.x = b for x using conjugate gradient method."""
    precon = spla.LinearOperator(a.shape, spla.spilu(a).solve) if precondition else None
    x, info = spla.cg(a, b, M=precon)
    print(f"CG solver finished with exit code {info}.")
    return x


def sparse_solve_minres(a, b, precondition=False):
    """Solve linear equations a.x = b for x using minimal residual method."""
    precon = spla.LinearOperator(a.shape, spla.spilu(a).solve) if precondition else None
    x, info = spla.minres(a, b, M=precon)
    print(f"MINRES solver finished with exit code {info}.")
    return x


def xu_system(mat):
    """Return priority vector for skew-symmetric additive fuzzy preference relation.

    Modified from https://link.springer.com/article/10.1007/s00500-010-0662-3
    """
    n = np.shape(mat)[0]
    nan_mask = np.isnan(mat)
    u = nan_mask.sum(axis=1)
    np.fill_diagonal(nan_mask, True)
    xu_arr = csc_matrix(nan_mask, dtype='float')
    xu_arr.setdiag(n - u)
    c = 1 + np.nansum(mat, axis=1) / ((n - 1) * np.nanmax(mat))
    w = sparse_solve_minres(xu_arr, c, precondition=False)
    return w


def norm_shift_arr(arr):
    """Rescale array elements to spread over interval [0, 1]."""
    arr -= np.amin(arr)
    arr /= np.amax(arr)
    return arr


# Game mode.
mode = 'standard'
# Top 10000 players or random data dump?
dump_type = 'top'

if mode in ['catch', 'fruits']:
    mode = 'fruits'

beatmaps = np.load("beatmaps.npy", allow_pickle=True).tolist()
num_beatmaps = len(beatmaps)
beatmap_enum = list(enumerate(beatmaps))
dist_arr = np.memmap("dist_arr.npy", dtype='float', mode='r', shape=(num_beatmaps, num_beatmaps))

difficulty = norm_shift_arr(xu_system(dist_arr))
sorted_diffs = []
for index, beatmap in beatmap_enum:
    beatmap_name = beatmap[1][:-4] if beatmap[1] else ''
    diff_tuple = (beatmap[0], difficulty[index], beatmap[2], beatmap_name) if mode == 'standard' else (beatmap[0], difficulty[index], beatmap_name)
    sorted_diffs.append(diff_tuple)
sorted_diffs.sort(key=lambda diff: diff[1], reverse=True)
types = 'int, float, float, object' if mode == 'standard' else 'int, float, object'
sorted_diff_arr = np.array(sorted_diffs, dtype=np.dtype(types))
formats = ('%-7u', '%.15f', '%.2f', '%s') if mode == 'standard' else ('%-7u', '%.15f', '%s')
np.savetxt('difficulty_top_30threshold_unbiased.txt', sorted_diff_arr, fmt=formats, delimiter='\t', encoding='utf-8')
