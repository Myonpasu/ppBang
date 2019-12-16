from scipy.stats import beta


def moment_to_shape(samp_mean, samp_var):
    alpha_over_mean = samp_mean * (1 - samp_mean) / samp_var - 1
    shape_alpha = samp_mean * alpha_over_mean
    shape_beta = alpha_over_mean - shape_alpha
    return shape_alpha, shape_beta


def performance(accuracy, difficulty, map_mean, map_var):
    shape_alpha, shape_beta = moment_to_shape(map_mean, map_var)
    normalization = difficulty / beta.cdf(map_mean, shape_alpha, shape_beta)
    pp = normalization * beta.cdf(accuracy, shape_alpha, shape_beta)
    return pp


def total_pp(pp_list):
    pp_list.sort(reverse=True)
    total = 0
    factor = 1
    for pp in pp_list:
        total += factor * pp
        factor *= 0.95
    return total
