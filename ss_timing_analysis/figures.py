import os

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import numpy as np
import scipy.stats

import veusz.embed

import figutils

import ss_timing_analysis.conf
import ss_timing_analysis.group_data
import ss_timing_analysis.group_fit
import ss_timing_analysis.dem


def sim_scatter(save_pdf=False):

    conf = ss_timing_analysis.conf.get_conf()

    # load the fit parameters, excluding the bad subjects
    # this will be subj x onsets x oris x (a,b) x (est, 2.5, 97.5)
    (fit, _, _) = ss_timing_analysis.group_fit.load_fit_data(
        exclude=True
    )

    # restrict to just the alpha estimates for the simultaneous condition
    data = fit[:, 1, :, 0, 0]

    # now convert to the suppression effect index; parallel - orthogonal
    data = data[:, 1] - data[:, 0]

    dem = ss_timing_analysis.dem.demographics()

    # righto, now for the sz scores
    sz = np.array(
        [
            dem[subj_id]["olife_total"]
            for subj_id in conf.subj_ids
        ]
    )

    # check that we've exlcuded subjects, as we think we should have
    assert len(sz) == len(data)

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")
    page.width.val = "18cm"
    page.height.val = "8cm"

    grid = page.Add("grid")
    grid.rows.val = 1
    grid.columns.val = 2

    grid.leftMargin.val = grid.rightMargin.val = "0cm"
    grid.bottomMargin.val = grid.topMargin.val = "0cm"

    grid.scaleCols.val = [0.85, 0.15]

    # SCATTER
    graph = grid.Add("graph", autoadd=False)
    graph.bottomMargin.val = "1cm"

    x_axis = graph.Add("axis")
    y_axis = graph.Add("axis")

    xy = graph.Add("xy")

    xy.xData.val = sz
    xy.yData.val = data
    xy.PlotLine.hide.val = True
    xy.MarkerFill.transparency.val = 60
    xy.MarkerLine.hide.val = True

    x_axis.label.val = "Schizotypy score"
    y_axis.label.val = "Context effect for simultaneous (par - orth)"
    y_axis.max.val = 0.5


    # KDE
    graph = grid.Add("graph", autoadd=False)

    graph.leftMargin.val = "0cm"
    graph.bottomMargin.val = "1cm"

    x_axis = graph.Add("axis")
    y_axis = graph.Add("axis")

    kde = scipy.stats.gaussian_kde(data)

    kde_x = np.linspace(0, 0.5, 100)
    kde_y = kde(kde_x)

    xy = graph.Add("xy")

    xy.xData.val = kde_y
    xy.yData.val = kde_x
    xy.MarkerFill.hide.val = True
    xy.MarkerLine.hide.val = True
    xy.FillBelow.fillto.val = "left"
    xy.FillBelow.color.val = "grey"
    xy.PlotLine.color.val = "grey"
    xy.FillBelow.hide.val = False

    y_axis.max.val = 0.5
    x_axis.hide.val = True
    x_axis.lowerPosition.val = 0.075

    if save_pdf:

        pdf_path = os.path.join(
            conf.figures_path,
            "ss_timing_sim_scatter.pdf"
        )

        embed.Export(pdf_path)

        log.info("Saving " + pdf_path + "...")

        (stem, _) = os.path.splitext(pdf_path)

        vsz_path = stem + ".vsz"

        embed.Save(vsz_path)

        log.info("Saving " + vsz_path + "...")

    embed.EnableToolbar(True)
    embed.WaitForClose()


def thresholds(save_pdf=False):

    conf = ss_timing_analysis.conf.get_conf()

    # load the fit parameters, excluding the bad subjects
    # this will be subj x onsets x oris x (a,b) x (est, 2.5, 97.5)
    (fit, _, _) = ss_timing_analysis.group_fit.load_fit_data(
        exclude=True
    )

    # restrict to just the alpha estimates
    data = fit[..., 0, 0]

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")
    page.width.val = "18cm"
    page.height.val = "8cm"

    # separate columns for simultaneous and leading
    grid = page.Add("grid")
    grid.rows.val = 1
    grid.columns.val = 2

    grid.leftMargin.val = grid.rightMargin.val = "0cm"
    grid.topMargin.val = grid.bottomMargin.val = "0cm"

    onset_order = [1, 0]  # sim, leading
    ori_order = [0, 1]  # orth, para

    for i_onset in onset_order:

        graph = grid.Add("graph", autoadd=False)
        graph.bottomMargin.val = "1cm"
        graph.topMargin.val = "0.6cm"

        x_axis = graph.Add("axis")
        y_axis = graph.Add("axis")

        for i_ori in ori_order:

            curr_data = data[:, i_onset, i_ori]

            boxplot = graph.Add("boxplot")

            dataset_str = "data_{onset:d}_{ori:d}".format(
                onset=i_onset,
                ori=i_ori
            )

            embed.SetData(
                dataset_str,
                curr_data
            )

            boxplot.values.val = dataset_str
            boxplot.posn.val = i_ori
            boxplot.labels.val = conf.surr_ori_labels[i_ori]
            boxplot.fillfraction.val = 0.3
            boxplot.markerSize.val = "2pt"

        for i_subj in xrange(conf.n_subj):

            subj_data_str = "subj_data_{subj:d}_{onset:d}".format(
                subj=i_subj, onset=i_onset
            )

            embed.SetData(
                subj_data_str,
                data[i_subj, i_onset, :]
            )

            xy = graph.Add("xy")

            xy.xData.val = [0, 1]
            xy.yData.val = subj_data_str
            xy.MarkerFill.hide.val = True
            xy.MarkerLine.hide.val = True
            xy.PlotLine.transparency.val = 80

        x_axis.mode.val = "labels"
        x_axis.MajorTicks.manualTicks.val = [0, 1]
        x_axis.MinorTicks.hide.val = True
        x_axis.label.val = "Relative orientation"

        y_axis.log.val = True
        y_axis.TickLabels.format.val = "%.3g"
        y_axis.MajorTicks.manualTicks.val = [0.001, 0.01, 0.1, 0.5, 1]
        y_axis.min.val = 0.005
        y_axis.max.val = 1.0
        y_axis.label.val = "Threshold contrast (%)"

        cond_label = graph.Add("label")

        cond_label.label.val = conf.surr_onset_labels[i_onset]
        cond_label.yPos.val = 1.02
        cond_label.xPos.val = 0.5
        cond_label.alignHorz.val = "centre"
        cond_label.Text.size.val = "10pt"

    if save_pdf:

        pdf_path = os.path.join(
            conf.figures_path,
            "ss_timing_thresholds.pdf"
        )

        embed.Export(pdf_path)

        log.info("Saving " + pdf_path + "...")

        (stem, _) = os.path.splitext(pdf_path)

        vsz_path = stem + ".vsz"

        embed.Save(vsz_path)

        log.info("Saving " + vsz_path + "...")

    embed.EnableToolbar(True)
    embed.WaitForClose()


def eg_subject(subj_id="p1022", save_pdf=False):

    conf = ss_timing_analysis.conf.get_conf()

    # find the index for this subject
    # we'll load the data without excluding participants, so this needs to be
    # in the all subjects list
    i_subj = conf.all_subj_ids.index(subj_id)

    # subj x onsets x oris x bins x (prop, n)
    data = ss_timing_analysis.group_data.bin_group_data()

    # restrict to this subject
    data = data[i_subj, ...]

    # fit is: subj x onsets x oris x (a, b) x (est, 2.5, 97.5)
    # fit_fine is: subj x onsets x oris x X x 2 (2.5, 97.5)
    (fit, fit_fine, _) = ss_timing_analysis.group_fit.load_fit_data()

    # again, restrict to this subject
    fit = fit[i_subj, ...]
    fit_fine = fit_fine[i_subj, ...]

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    embed.SetData("bin_centres", conf.bin_centres)
    embed.SetData("fine_x", conf.fine_x)

    page = embed.Root.Add("page")

    page.width.val = "8cm"
    page.height.val = "20cm"

    grid = page.Add("grid")

    grid.rows.val = 4
    grid.columns.val = 1

    grid.leftMargin.val = grid.rightMargin.val = "0cm"
    grid.topMargin.val = grid.bottomMargin.val = "0cm"

    onset_order = [1, 0]  # sim, leading
    ori_order = [0, 1]  # orth, par

    for i_onset in onset_order:
        for i_ori in ori_order:

            graph = grid.Add("graph", autoadd=False)
            graph.leftMargin.val = "1.2cm"
            graph.rightMargin.val = "0.5cm"
            graph.topMargin.val = "0.7cm"
            graph.bottomMargin.val = "0.85cm"

            x_axis = graph.Add("axis")
            y_axis = graph.Add("axis")

            cond_label = graph.Add("label")

            cond_label.label.val = ", ".join(
                [
                    conf.surr_ori_labels[i_ori],
                    conf.surr_onset_labels[i_onset]
                ]
            )
            cond_label.yPos.val = 1.02
            cond_label.xPos.val = 0.5
            cond_label.alignHorz.val = "centre"
            cond_label.Text.size.val = "10pt"

            # CROSSHAIRS
            pse_y = graph.Add("xy")
            pse_y.xData.val = [0.001, fit[i_onset, i_ori, 0, 0]]
            pse_y.yData.val = [1 - np.exp(-1) + 0.04] * 2

            pse_x = graph.Add("xy")
            pse_x.xData.val = [fit[i_onset, i_ori, 0, 0]] * 2
            pse_x.yData.val = [-0.05, 1 - np.exp(-1) + 0.04]

            for pse_ax in (pse_y, pse_x):
                pse_ax.MarkerFill.hide.val = True
                pse_ax.MarkerLine.hide.val = True
                pse_ax.PlotLine.style.val = "dashed"


            # POINTS
            points = graph.Add("xy")

            prop_name = "resp_prop_{t:d}_{o:d}".format(
                t=i_onset, o=i_ori
            )

            embed.SetData(
                prop_name,
                data[i_onset, i_ori, :, 0]
            )

            k_name = "resp_k_{t:d}_{o:d}".format(
                t=i_onset, o=i_ori
            )

            point_scale = np.sqrt(
                (data[i_onset, i_ori, :, 1] * 1) / np.pi
            ) * 2 * 0.35

            embed.SetData(
                k_name,
                point_scale
            )

            points.xData.val = "bin_centres"
            points.yData.val = prop_name

            points.scalePoints.val = k_name

            points.MarkerLine.hide.val = True
            points.MarkerFill.transparency.val = 65
            points.PlotLine.hide.val = True
            points.MarkerFill.color.val = "blue"

            # FIT
            fit_plot = graph.Add("xy")

            fit_name = "fit_{t:d}_{o:d}".format(
                t=i_onset, o=i_ori
            )

            fit_y = conf.psych_func(
                conf.fine_x,
                alpha=fit[i_onset, i_ori, 0, 0],
                beta=fit[i_onset, i_ori, 1, 0]
            )

            embed.SetData(
                fit_name,
                fit_y,
                poserr=np.abs(
                    fit_fine[i_onset, i_ori, :, 1] - fit_y
                ),
                negerr=np.abs(
                    fit_fine[i_onset, i_ori, :, 0] - fit_y
                )
            )

            fit_plot.xData.val = "fine_x"
            fit_plot.yData.val = fit_name

            fit_plot.MarkerFill.hide.val = True
            fit_plot.MarkerLine.hide.val = True
            fit_plot.errorStyle.val = "fillvert"
            fit_plot.ErrorBarLine.hide.val = True

            x_axis.log.val = True
            x_axis.label.val = "Target contrast"
            x_axis.TickLabels.format.val = "%.3g"
            x_axis.MajorTicks.manualTicks.val = [0.001, 0.01, 0.1, 0.5, 1]

            y_axis.min.val = -0.1
            y_axis.max.val = 1.1
            y_axis.label.val = "Accuracy (prop. correct)"
            y_axis.MinorTicks.hide.val = True
            y_axis.MajorTicks.manualTicks.val = [0, 0.25, 0.5, 0.67, 1]
            y_axis.TickLabels.format.val = "%.02f"

    if save_pdf:

        pdf_path = os.path.join(
            conf.figures_path,
            "ss_timing_eg_subject.pdf"
        )

        embed.Export(pdf_path)

        log.info("Saving " + pdf_path + "...")

        (stem, _) = os.path.splitext(pdf_path)

        vsz_path = stem + ".vsz"

        embed.Save(vsz_path)

        log.info("Saving " + vsz_path + "...")

    embed.EnableToolbar(True)
    embed.WaitForClose()


def subjects(save_pdf=False):

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
                cond_label.label.val = ", ".join(
                    [
                        conf.surr_ori_labels[i_ori],
                        conf.surr_onset_labels[i_onset]
                    ]
                )
                cond_label.yPos.val = 1.02
                cond_label.xPos.val = 0.5
                cond_label.alignHorz.val = "centre"
                cond_label.Text.size.val = "10pt"

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
                    data[i_subj, i_onset, i_ori, :, 1] / np.pi
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

                y_axis.min.val = -0.1
                y_axis.max.val = 1.1
                y_axis.label.val = "Accuracy (prop. correct)"
                y_axis.MinorTicks.hide.val = True
                y_axis.MajorTicks.manualTicks.val = [0, 0.25, 0.5, 0.67, 1]
                y_axis.TickLabels.format.val = "%.02f"

    if save_pdf:

        pdf_path = os.path.join(
            conf.figures_path,
            "ss_timing_subjects.pdf"
        )

        embed.Export(pdf_path, page=range(conf.n_all_subj))

        log.info("Saving " + pdf_path + "...")

        (stem, _) = os.path.splitext(pdf_path)

        vsz_path = stem + ".vsz"

        embed.Save(vsz_path)

        log.info("Saving " + vsz_path + "...")

    embed.EnableToolbar(True)
    embed.WaitForClose()

