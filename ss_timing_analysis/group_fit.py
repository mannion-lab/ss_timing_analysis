import os

import numpy as np
import scipy.stats

import ss_timing_analysis.conf
import ss_timing_analysis.group_data
import ss_timing_analysis.fit


def fit_data():

    conf = ss_timing_analysis.conf.get_conf()

    data = ss_timing_analysis.group_data.load_group_data()

    fit_params = np.empty(
        (
            conf.n_all_subj,
            conf.n_surr_onsets,
            conf.n_surr_oris,
            2,  # (alpha, beta)
            3  # (est, 2.5%, 97.5%)
        )
    )
    fit_params.fill(np.NAN)

    fit_fine_boot = np.empty(
        (
            conf.n_all_subj,
            conf.n_surr_onsets,
            conf.n_surr_oris,
            conf.n_fine_x,
            conf.n_boot
        )
    )
    fit_fine_boot.fill(np.NAN)

    boot_all = np.empty(
        (
            conf.n_all_subj,
            conf.n_surr_onsets,
            conf.n_surr_oris,
            2,
            conf.n_boot
        )
    )
    boot_all.fill(np.NAN)

    np.random.seed(conf.boot_seed)

    for i_subj in xrange(conf.n_all_subj):

        print "{n:d} / {t:d}".format(n=i_subj + 1, t=conf.n_all_subj)

        boot_data = ss_timing_analysis.fit.fit_ml_boot_subj(
            data=data[i_subj, ...],
            conf=conf,
            n_boot=conf.n_boot
        )

        boot_all[i_subj, ...] = boot_data[..., 1:]

        # first is without bootstrapping
        fit_params[i_subj, ..., 0] = boot_data[..., 0]

        fit_cis = scipy.stats.scoreatpercentile(
            boot_data[..., 1:],  # dont include non-booted
            [2.5, 97.5],
            axis=-1
        )

        fit_params[i_subj, ..., 1] = fit_cis[0, ...]
        fit_params[i_subj, ..., 2] = fit_cis[1, ...]

        for i_boot in xrange(conf.n_boot):
            for i_onset in xrange(conf.n_surr_onsets):
                for i_ori in xrange(conf.n_surr_oris):

                    (a, b) = boot_data[i_onset, i_ori, :, i_boot + 1]

                    pred = conf.psych_func(x=conf.fine_x, alpha=a, beta=b)

                    fit_fine_boot[i_subj, i_onset, i_ori, :, i_boot] = pred

    fit_fine = np.empty(list(fit_fine_boot.shape[:-1]) + [2])
    fit_fine.fill(np.NAN)

    fit_fine_cis = scipy.stats.scoreatpercentile(
        fit_fine_boot,
        [2.5, 97.5],
        axis=-1
    )

    fit_fine[..., 0] = fit_fine_cis[0, ...]
    fit_fine[..., 1] = fit_fine_cis[1, ...]

    npz_path = os.path.join(
        conf.group_data_path,
        "ss_timing_data_fits.npz"
    )

    np.savez(
        npz_path,
        fit_params=fit_params,
        fit_fine=fit_fine,
        boot_all=boot_all
    )

    return (fit_params, fit_fine, boot_all)


def load_fit_data(exclude=False):

    conf = ss_timing_analysis.conf.get_conf()

    npz_path = os.path.join(
        conf.group_data_path,
        "ss_timing_data_fits.npz"
    )

    fit = np.load(npz_path)

    fit_params = fit["fit_params"]
    fit_fine = fit["fit_fine"]
    fit_boot = fit["boot_all"]

    if exclude:

        # work out which indices correspond to the bad subjects
        i_bad = [
            conf.all_subj_ids.index(bad_subj_id)
            for bad_subj_id in conf.exclude_ids
        ]

        # first, just check that this does what we think it does
        all_subj_ids = np.array(conf.all_subj_ids)
        subj_ids = np.delete(all_subj_ids, i_bad)
        assert list(subj_ids) == conf.subj_ids

        # also check that our manual conf exclusions are the same as calculated
        # from the data
        assert (
            conf.exclude_ids == group_fit_exclusions()
        )

        # righto, now actually do it

        fit_params = np.delete(
            fit_params,
            i_bad,
            axis=0
        )

        fit_fine = np.delete(
            fit_fine,
            i_bad,
            axis=0
        )

        fit_boot = np.delete(
            fit_boot,
            i_bad,
            axis=0
        )

    return (fit_params, fit_fine, fit_boot)


def group_fit_exclusions():

    conf = ss_timing_analysis.conf.get_conf()

    (fit, _, _) = load_fit_data()

    # just look at the slope parameters
    fit = fit[..., 1, 0]

    bad_ids = set()

    for i_onset in xrange(conf.n_surr_onsets):
        for i_ori in xrange(conf.n_surr_oris):

            cutoff = (
                np.mean(fit[:, i_onset, i_ori]) -
                2 * np.std(fit[:, i_onset, i_ori], ddof=1)
            )

            i_bad = np.argwhere(fit[:, i_onset, i_ori] < cutoff)

            bad_ids.update(
                [
                    conf.all_subj_ids[i_curr_bad]
                    for i_curr_bad in i_bad
                ]
            )

    bad_ids = list(bad_ids)
    bad_ids.sort()

    return bad_ids








