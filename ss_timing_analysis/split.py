
import numpy as np
import scipy.stats

import ss_timing_analysis.conf
import ss_timing_analysis.dem
import ss_timing_analysis.group_fit


def split_ana():

    conf = ss_timing_analysis.conf.get_conf()

    split_locs = [50, 40, 30, 20, 10]
    n_splits = len(split_locs)

    fit_data = ss_timing_analysis.group_fit.load_fit_data(exclude=True)

    alphas = fit_data[0][:, 1, :, 0, 0]

    ceff = alphas[:, 1] - alphas[:, 0]

    subscales = {
        subscale: ss_timing_analysis.dem.get_olife_subscale(
            subscale=subscale, exclude=True
        )
        for subscale in conf.subscales
    }

    data = np.full((n_splits, len(conf.subscales), 2), np.nan)

    for (i_split, split_loc) in enumerate(split_locs):

        for (i_sub, subscale_name) in enumerate(conf.subscales):

            olife = np.array(subscales[subscale_name])

            i_top = np.where(
                olife >= scipy.stats.scoreatpercentile(olife, 100 - split_loc)
            )[0]

            i_bottom = np.where(
                olife < scipy.stats.scoreatpercentile(olife, split_loc)
            )[0]

            top_ceff = ceff[i_top]
            top_mean = np.mean(top_ceff)
            top_n = len(top_ceff)
            top_std = np.std(top_ceff, ddof=1)

            bottom_ceff = ceff[i_bottom]
            bottom_mean = np.mean(bottom_ceff)
            bottom_n = len(bottom_ceff)
            bottom_std = np.std(bottom_ceff, ddof=1)

            mean_diff = top_mean - bottom_mean
            sem_diff = np.sqrt(
                (
                    (top_std ** 2) / top_n
                ) +
                (
                    (bottom_std ** 2) / bottom_n
                )
            )

            data[i_split, i_sub, :] = (mean_diff, sem_diff)

            print scipy.stats.ttest_ind(top_ceff, bottom_ceff)

    return data
