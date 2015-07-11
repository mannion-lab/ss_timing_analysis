
import numpy as np

import matplotlib.pyplot as plt

import figutils
figutils.set_defaults()

import ss_timing_analysis.conf
import ss_timing_analysis.analysis


def plot_subj_fits(subj_id):

    conf = ss_timing_analysis.conf.get_conf(subj_id)

    # onsets x oris x trials x [contrast, resp]
    resp_data = ss_timing_analysis.analysis.get_subj_resp_data(conf)

    # onsets x oris x [alpha, beta]
    fit_params = ss_timing_analysis.analysis.get_fit_params(conf)

    min_c = np.min(resp_data[..., 0])
    max_c = np.max(resp_data[..., 1])

    if min_c < 0.006:
        min_c = 0.006
    if max_c > 0.95:
        max_c = 0.95

    min_c = min_c - 0.005
    max_c = max_c + 0.005

    fine_x = np.logspace(
        np.log10(min_c),
        np.log10(max_c),
        100
    )

    n_bins = 20
    bins = np.logspace(
        np.log10(min_c),
        np.log10(max_c),
        n_bins
    )

    fig = plt.figure(figsize=[15.5, 11.4])

    ax_k = 1

    for i_onset in xrange(conf.n_surr_onsets):
        for i_ori in xrange(conf.n_surr_oris):

            data = resp_data[i_onset, i_ori, :]

            i_bins = np.digitize(data[:, 0], bins)

            counts = np.zeros(n_bins)
            totals = np.zeros(n_bins)
            resp_p = np.zeros(n_bins)

            for i_bin in xrange(n_bins):

                in_bin = (i_bin == i_bins)

                n_in_bin = np.sum(in_bin)

                n_corr_in_bin = np.sum(data[in_bin, 1])

                counts[i_bin] = n_corr_in_bin
                totals[i_bin] = n_in_bin

                try:
                    resp_p[i_bin] = float(n_corr_in_bin) / n_in_bin
                except ZeroDivisionError:
                    pass

            ax = plt.subplot(
                conf.n_surr_onsets,
                conf.n_surr_oris,
                ax_k
            )

            ax.hold(True)

            ax.scatter(bins, resp_p, s=totals)

            pred_y = conf.psych_func(
                fine_x,
                fit_params[i_onset, i_ori, 0],
                fit_params[i_onset, i_ori, 1]
            )

            ax.plot(fine_x, pred_y)

            ax.set_xscale("log")

            ax.set_xlim([min_c, max_c])
            ax.set_ylim([-0.05, 1.05])

            ax.set_xlabel("Contrast")
            ax.set_ylabel("Accuracy (% correct)")

            xticks = ax.get_xticks()

            xtick_labels = [
                "{n:.4g}".format(n=tick)
                for tick in xticks
            ]

            ax.set_xticklabels(xtick_labels)

            title = (
                "Onset: " + conf.surr_onsets[i_onset] +
                ", Ori: " + conf.surr_ori_labels[i_ori]
            )

            ax.set_title(title)

            figutils.cleanup_fig(ax)

            ax_k += 1

    fig.subplots_adjust(
        top=0.97,
        bottom=0.05,
        left=0.03,
        right=0.98,
        hspace=0.23,
        wspace=0.19
    )

def plot_subj_conds(subj_id):

    conf = ss_timing_analysis.conf.get_conf(subj_id)

    # onsets x oris x [alpha, beta]
    fit_params = ss_timing_analysis.analysis.get_fit_params(conf)

    cols = ["b", "g"]

    fig = plt.figure()

    ax = plt.subplot(1, 1, 1)
    ax.hold(True)

    for i_onset in xrange(conf.n_surr_onsets):

        ax.plot(
            [0, 1],
            fit_params[i_onset, :, 0],
            c=cols[i_onset],
            label=conf.surr_onsets[i_onset]
        )

        ax.scatter(
            [0, 1],
            fit_params[i_onset, :, 0],
            edgecolor=[1] * 3,
            facecolor=cols[i_onset],
            s=80,
            label="_nolegend"
        )

    ax.set_xlim([-0.25, 1.25])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(conf.surr_ori_labels)

    ax.set_xlabel("Surround orientation")
    ax.set_ylabel("Detection threshold")

    ax.legend(loc="upper left")

    figutils.cleanup_fig(ax)

