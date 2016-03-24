
import numpy as np
import scipy.stats

import ss_timing_analysis.conf
import ss_timing_analysis.group_fit
import ss_timing_analysis.dem


def correlations(conf=None):

    if conf is None:
        conf = ss_timing_analysis.conf.get_conf()

    corr_func = scipy.stats.spearmanr

    all_r_p = np.empty(
        (
            3,  # analyses
            len(conf.subscales),
            2
        )
    )
    all_r_p.fill(np.nan)

    r_boot = np.empty(
        (
            3,
            len(conf.subscales),
            conf.n_boot
        )
    )
    r_boot.fill(np.NAN)

    np.random.seed(conf.boot_seed)

    # n_subj array
    olife_total = ss_timing_analysis.dem.get_olife_total(exclude=True)

    # n_subj x onset x ori x (a, b) x (est, ci...)
    (data, _, _) = ss_timing_analysis.group_fit.load_fit_data(exclude=True)

    n_subj = data.shape[0]

    # restrict to the alpha estimates
    data = data[..., 0, 0]

    #--------
    # 1. simultaneous, SS effect x olife

    # sim, par - sim, orth
    sim_ss = data[:, 1, 1] - data[:, 1, 0]

    for (i_sub, subscale) in enumerate(conf.subscales):

        sub_total = ss_timing_analysis.dem.get_olife_subscale(
            subscale,
            exclude=True
        )

        print "Simultaneous, SS x O-LIFE {s:s}:".format(s=subscale)

        (r, p) = corr_func(sub_total, sim_ss)

        print "\tr({n:d}) = {r:.4f}, p = {p:.4f}".format(
            n=len(sub_total) - 2, r=r, p=p
        )

        all_r_p[0, i_sub, :] = (r, p)

        # bootstrappin'
        for i_boot in xrange(conf.n_boot):

            i_subj = np.random.choice(
                a=range(n_subj),
                size=n_subj,
                replace=True
            )

            (curr_boot_r, _) = corr_func(
                sub_total[i_subj],
                sim_ss[i_subj]
            )

            r_boot[0, i_sub, i_boot] = curr_boot_r

        ci = scipy.stats.scoreatpercentile(
            r_boot[0, i_sub, :],
            [2.5, 97.5]
        )

        print "\tCI = [{a:.3f}, {b:.3f}]".format(
            a=ci[0], b=ci[1]
        )

    print "-" * 10

    #--------
    # 2. simultaneous, orth x olife

    # sim, orth
    sim_orth = data[:, 1, 0]

    for (i_sub, subscale) in enumerate(conf.subscales):

        sub_total = ss_timing_analysis.dem.get_olife_subscale(
            subscale,
            exclude=True
        )

        print "Simultaneous, orth x O-LIFE {s:s}:".format(s=subscale)

        (r, p) = corr_func(sub_total, sim_orth)

        print "\tr({n:d}) = {r:.4f}, p = {p:.4f}".format(
            n=len(sub_total) - 2, r=r, p=p
        )

        all_r_p[1, i_sub, :] = (r, p)

        # bootstrappin'
        for i_boot in xrange(conf.n_boot):

            i_subj = np.random.choice(
                a=range(n_subj),
                size=n_subj,
                replace=True
            )

            (curr_boot_r, _) = corr_func(
                sub_total[i_subj],
                sim_orth[i_subj]
            )

            r_boot[1, i_sub, i_boot] = curr_boot_r

        ci = scipy.stats.scoreatpercentile(
            r_boot[1, i_sub, :],
            [2.5, 97.5]
        )

        print "\tCI = [{a:.3f}, {b:.3f}]".format(
            a=ci[0], b=ci[1]
        )

    print "-" * 10

    #---------
    # 3. (delay, par) - (delay, orth) x olife
    ss_delay = data[:, 0, 1] - data[:, 0, 0]

    for (i_sub, subscale) in enumerate(conf.subscales):

        sub_total = ss_timing_analysis.dem.get_olife_subscale(
            subscale,
            exclude=True
        )

        print "SS @ delay x O-LIFE {s:s}:".format(s=subscale)

        (r, p) = corr_func(sub_total, ss_delay)

        print "\tr({n:d}) = {r:.4f}, p = {p:.4f}".format(
            n=len(sub_total) - 2, r=r, p=p
        )

        all_r_p[2, i_sub, :] = (r, p)

        # bootstrappin'
        for i_boot in xrange(conf.n_boot):

            i_subj = np.random.choice(
                a=range(n_subj),
                size=n_subj,
                replace=True
            )

            (curr_boot_r, _) = corr_func(
                sub_total[i_subj],
                ss_delay[i_subj]
            )

            r_boot[2, i_sub, i_boot] = curr_boot_r

        ci = scipy.stats.scoreatpercentile(
            r_boot[2, i_sub, :],
            [2.5, 97.5]
        )

        print "\tCI = [{a:.3f}, {b:.3f}]".format(
            a=ci[0], b=ci[1]
        )

    assert np.sum(np.isnan(all_r_p)) == 0
    assert np.sum(np.isnan(r_boot)) == 0

    return all_r_p


def descriptives():

    conf = ss_timing_analysis.conf.get_conf()

    # n_subj x onset x ori x (a, b) x (est, ci...)
    (data, _, _) = ss_timing_analysis.group_fit.load_fit_data(exclude=True)

    # restrict to the alpha estimates
    data = data[..., 0, 0]

    # convert to percent
    data *= 100.0

    mean = np.mean(data, axis=0)
    sem = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])

    n_subj = data.shape[0]

    np.random.seed(conf.boot_seed)

    for i_onset in xrange(conf.n_surr_onsets):
        for i_ori in xrange(conf.n_surr_oris):

            boots = np.empty((conf.n_boot))
            boots.fill(np.NAN)

            for i_boot in xrange(conf.n_boot):

                i_subj = np.random.choice(
                    a=range(n_subj),
                    size=n_subj,
                    replace=True
                )

                boots[i_boot] = np.mean(data[i_subj, i_onset, i_ori])

            ci = scipy.stats.scoreatpercentile(boots, [2.5, 97.5])

            print "{t:s}, {o:s} = {m:.4f}, CI = [{a:.3f}, {b:.3f}]".format(
                t=conf.surr_onset_labels[i_onset],
                o=conf.surr_ori_labels[i_ori],
                m=mean[i_onset, i_ori],
                a=ci[0], b=ci[1]
            )
