
import numpy as np

import veusz.embed

import figutils

import ss_timing_analysis.conf
import ss_timing_analysis.group_data
import ss_timing_analysis.group_fit


def subjects(save_pdf=True):

    conf = ss_timing_analysis.conf.get_conf()

    # subj x onsets x oris x bins x (prop, n)
    data = ss_timing_analysis.group_data.bin_group_data()

    # fit is: subj x onsets x oris x (a, b) x (est, 2.5, 97.5)
    # fit_fine is: subj x onsets x oris x X x 2 (2.5, 97.5)
    (fit, fit_fine, _) = ss_timing_analysis.group_fit.load_fit_data()

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    embed.SetData("bin_centres", conf.bin_centres)
    embed.SetData("fine_x", conf.fine_x)

    for (i_subj, subj_id) in enumerate(conf.all_subj_ids):

        page = embed.Root.Add("page")

        page.width.val = "15cm"
        page.height.val = "15cm"

        label = page.Add("label")

        label.label.val = subj_id
        label.yPos.val = 0.95

        if subj_id in conf.exclude_ids:
            label.label.val += " (excluded)"
            label.Text.color.val = "red"

        grid = page.Add("grid")

        grid.rows.val = 2
        grid.columns.val = 2

        grid.leftMargin.val = grid.rightMargin.val = "0cm"
        grid.topMargin.val = "1.2cm"
        grid.bottomMargin.val = "0cm"

        for i_onset in xrange(conf.n_surr_onsets):
            for i_ori in xrange(conf.n_surr_oris):

                graph = grid.Add("graph", autoadd=False)

                x_axis = graph.Add("axis")
                y_axis = graph.Add("axis")

                cond_label = graph.Add("label")

                cond_label.label.val = (
                    "Onset: " + conf.surr_onsets[i_onset] + "; " +
                    "Ori: " + conf.surr_ori_labels[i_ori]
                )
                cond_label.yPos.val = 1.02
                cond_label.xPos.val = 0.22

                # CROSSHAIRS
                pse_y = graph.Add("xy")
                pse_y.xData.val = [0.001, fit[i_subj, i_onset, i_ori, 0, 0]]
                pse_y.yData.val = [1 - np.exp(-1) + 0.04] * 2

                pse_x = graph.Add("xy")
                pse_x.xData.val = [fit[i_subj, i_onset, i_ori, 0, 0]] * 2
                pse_x.yData.val = [-0.05, 1 - np.exp(-1) + 0.04]

                for pse_ax in (pse_y, pse_x):
                    pse_ax.MarkerFill.hide.val = True
                    pse_ax.MarkerLine.hide.val = True
                    pse_ax.PlotLine.style.val = "dashed"


                # POINTS
                points = graph.Add("xy")

                prop_name = "resp_prop_{s:d}_{t:d}_{o:d}".format(
                    s=i_subj, t=i_onset, o=i_ori
                )

                embed.SetData(
                    prop_name,
                    data[i_subj, i_onset, i_ori, :, 0]
                )

                k_name = "resp_k_{s:d}_{t:d}_{o:d}".format(
                    s=i_subj, t=i_onset, o=i_ori
                )

                point_scale = np.sqrt(
                    (data[i_subj, i_onset, i_ori, :, 1] * 1) / np.pi
                ) * 2 * 0.35

                embed.SetData(
                    k_name,
                    point_scale
                )

                points.xData.val = "bin_centres"
                points.yData.val = prop_name

                points.scalePoints.val = k_name

                points.MarkerLine.hide.val = True
                points.MarkerFill.transparency.val = 50
                points.PlotLine.hide.val = True
                points.MarkerFill.color.val = "blue"

                # FIT
                fit_plot = graph.Add("xy")

                fit_name = "fit_{s:d}_{t:d}_{o:d}".format(
                    s=i_subj, t=i_onset, o=i_ori
                )

                fit_y = conf.psych_func(
                    conf.fine_x,
                    alpha=fit[i_subj, i_onset, i_ori, 0, 0],
                    beta=fit[i_subj, i_onset, i_ori, 1, 0]
                )

                embed.SetData(
                    fit_name,
                    fit_y,
                    poserr=np.abs(
                        fit_fine[i_subj, i_onset, i_ori, :, 1] - fit_y
                    ),
                    negerr=np.abs(
                        fit_fine[i_subj, i_onset, i_ori, :, 0] - fit_y
                    )
                )

                fit_plot.xData.val = "fine_x"
                fit_plot.yData.val = fit_name

                fit_plot.MarkerFill.hide.val = True
                fit_plot.MarkerLine.hide.val = True
                fit_plot.errorStyle.val = "fillvert"
                fit_plot.ErrorBarLine.hide.val = True

                x_axis.log.val = True
                x_axis.label.val = "Contrast"
                x_axis.TickLabels.format.val = "%.3g"
                x_axis.MajorTicks.manualTicks.val = [0.001, 0.01, 0.1, 0.5, 1]

                y_axis.min.val = -0.05
                y_axis.max.val = 1.05
                y_axis.label.val = "Accuracy (%)"


    if save_pdf is False:
        embed.EnableToolbar(True)
        embed.WaitForClose()
        embed.Save("/home/damien/ss.vsz")
    else:
        embed.Export("/home/damien/ss_1000.pdf", page=range(conf.n_all_subj))
        embed.Close()


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

    n_bins = 50
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

    ax.set_yscale("log")

    ax.set_xlabel("Surround orientation")
    ax.set_ylabel("Detection threshold")

    ax.legend(loc="upper left")

    figutils.cleanup_fig(ax)

