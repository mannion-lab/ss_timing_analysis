
import numpy as np
import scipy.stats

import ss_timing_analysis.conf
import ss_timing_analysis.group_fit
import ss_timing_analysis.dem


def correlations():

    corr_func = scipy.stats.spearmanr

    conf = ss_timing_analysis.conf.get_conf()

    # n_subj array
    olife_total = ss_timing_analysis.dem.get_olife_total()

    # n_subj x onset x ori x (a, b) x (est, ci...)
    (data, _, _) = ss_timing_analysis.group_fit.load_fit_data(exclude=True)

    # restrict to the alpha estimates
    data = data[..., 0, 0]

    #--------
    # 1. simultaneous, SS effect x olife

    # sim, par - sim, orth
    sim_ss = data[:, 1, 1] - data[:, 1, 0]

    print "Simultaneous, SS x O-LIFE total:"

    (r, p) = corr_func(olife_total, sim_ss)

    print "\tr({n:d}) = {r:.4f}, p = {p:.4f}".format(
        n=len(olife_total) - 2, r=r, p=p
    )

    #--------
    # 1a. subscale

    for subscale in ["int_anh", "imp_non", "cog_dis", "un_ex"]:

        sub_total = ss_timing_analysis.dem.get_olife_subscale(subscale)

        print "Simultaneous, SS x O-LIFE {s:s}:".format(s=subscale)

        (r, p) = corr_func(sub_total, sim_ss)

        print "\tr({n:d}) = {r:.4f}, p = {p:.4f}".format(
            n=len(sub_total) - 2, r=r, p=p
        )

    print "-" * 10

    #--------
    # 2. simultaneous, orth x olife

    # sim, orth
    sim_orth = data[:, 1, 0]

    print "Simultaneous, orth x O-LIFE total:"

    (r, p) = corr_func(olife_total, sim_orth)

    print "\tr({n:d}) = {r:.4f}, p = {p:.4f}".format(
        n=len(olife_total) - 2, r=r, p=p
    )

    #--------
    # 2a. subscale

    for subscale in ["int_anh", "imp_non", "cog_dis", "un_ex"]:

        sub_total = ss_timing_analysis.dem.get_olife_subscale(subscale)

        print "Simultaneous, orth x O-LIFE {s:s}:".format(s=subscale)

        (r, p) = corr_func(sub_total, sim_orth)

        print "\tr({n:d}) = {r:.4f}, p = {p:.4f}".format(
            n=len(sub_total) - 2, r=r, p=p
        )

    print "-" * 10

    #---------
    # 3. (sim, SS) - (delay, SS) x olife
    #ss_diff = (data[:, 1, 1] - data[:, 1, 0]) - (data[:, 0, 1] - data[:, 0, 0])
    ss_diff = data[:, 0, 1] - data[:, 0, 0]

#    print "(SS @ sim) - (SS @ delay) x O-LIFE total:"
    print "SS @ delay x O-LIFE total:"

    (r, p) = corr_func(olife_total, ss_diff)

    print "\tr({n:d}) = {r:.4f}, p = {p:.4f}".format(
        n=len(olife_total) - 2, r=r, p=p
    )

    #--------
    # 3a. subscale

    for subscale in ["int_anh", "imp_non", "cog_dis", "un_ex"]:

        sub_total = ss_timing_analysis.dem.get_olife_subscale(subscale)

        #print "(SS @ sim) - (SS @ delay) x O-LIFE {s:s}:".format(s=subscale)
        print "SS @ delay x O-LIFE {s:s}:".format(s=subscale)

        (r, p) = corr_func(sub_total, ss_diff)

        print "\tr({n:d}) = {r:.4f}, p = {p:.4f}".format(
            n=len(sub_total) - 2, r=r, p=p
        )



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

    for i_onset in xrange(conf.n_surr_onsets):
        for i_ori in xrange(conf.n_surr_oris):

            print "{t:s}, {o:s} = {m:.4f}, SE = {se:.4f}".format(
                t=conf.surr_onset_labels[i_onset],
                o=conf.surr_ori_labels[i_ori],
                m=mean[i_onset, i_ori],
                se=sem[i_onset, i_ori]
            )
