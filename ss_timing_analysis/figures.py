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

def _save(embed, conf, name_str, dpi=600, page=0, multi_dpi=150):

    for ext in [".pdf", ".png", ".tiff"]:

        out_path = os.path.join(
            conf.figures_path,
            name_str
        ) + ext

        if ext == ".pdf":
            embed.Export(out_path, page=page)

        else:

            try:
                embed.Export(out_path, dpi=dpi, backcolor="white", page=page)

            except RuntimeError as e:

                if "Can only export a single page in this format" in e.message:

                    for page_num in page:

                        page_out_path = os.path.join(
                            conf.figures_path,
                            name_str + "_{n:d}".format(n=page_num)
                        ) + ext

                        embed.Export(
                            page_out_path,
                            dpi=multi_dpi,
                            backcolor="white",
                            page=page_num
                        )

        log.info("Saving " + out_path + "...")

    (stem, _) = os.path.splitext(out_path)

    vsz_path = stem + ".vsz"

    embed.Save(vsz_path)

    log.info("Saving " + vsz_path + "...")


def scatter(cond, form, save_pdf=False):

    if cond not in ["sim", "sim_orth", "lead"]:
        raise ValueError()

    if form not in ["linear", "rank"]:
        raise ValueError()

    conf = ss_timing_analysis.conf.get_conf()

    # load the fit parameters, excluding the bad subjects
    # this will be subj x onsets x oris x (a,b) x (est, 2.5, 97.5)
    (fit, _, _) = ss_timing_analysis.group_fit.load_fit_data(exclude=True)

    # restrict to just the alpha estimates
    data = fit[..., 0, 0]

    if cond == "sim":
        data = data[:, 1, 1] - data[:, 1, 0]
    elif cond == "sim_orth":
        data = data[:, 1, 0]
    elif cond == "lead":
        data = (data[:, 1, 1] - data[:, 1, 0]) - (data[:, 0, 1] - data[:, 0, 0])

    if form == "rank":
        data = scipy.stats.rankdata(data)

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    embed.SetData("data", data)

    page = embed.Root.Add("page")

    page.width.val = "14cm"
    page.height.val = "10cm"

    grid = page.Add("grid")

    grid.rows.val = 2
    grid.columns.val = 2

    grid.leftMargin.val = grid.rightMargin.val = "2cm"
    grid.topMargin.val = grid.bottomMargin.val = "0cm"

    ss_nice = [
        "Unusual experiences",
        "Cognitive disorganisation",
        "Introvertive anhedonia",
        "Impulsive nonconformity"
    ]

    for (i_sub, subscale) in enumerate(
        ("un_ex", "cog_dis", "int_anh", "imp_non")
    ):

        curr_ss = ss_timing_analysis.dem.get_olife_subscale(
            subscale,
            exclude=True
        )

        if form == "rank":
            curr_ss = scipy.stats.rankdata(curr_ss)

        assert len(curr_ss) == len(data)

        graph = grid.Add("graph", autoadd=False)
        graph.bottomMargin.val = "1cm"
        graph.topMargin.val = "0.6cm"
        graph.leftMargin.val = "1cm"
        graph.aspect.val = 1

        label = graph.Add("label")

        label.label.val = ss_nice[i_sub]
        label.yPos.val = 1.025
        label.xPos.val = 0.5
        label.alignHorz.val = "centre"

        x_axis = graph.Add("axis")
        y_axis = graph.Add("axis")

        xy = graph.Add("xy")

        xy.xData.val = curr_ss
        xy.yData.val = data
        xy.PlotLine.hide.val = True
        xy.MarkerFill.transparency.val = 60
        xy.MarkerLine.hide.val = True
        xy.markerSize.val = "2pt"

        if form == "rank":
            x_axis.label.val = "O-LIFE subscale (rank; 1 = lowest)"
        else:
            x_axis.label.val = "O-LIFE subscale score"

        if cond == "sim":
            if form == "rank":
                y_axis.label.val = "Context effect (rank; 1 = lowest)"
            else:
                y_axis.label.val = "Context effect (contrast units)"

        elif cond == "sim_orth":
            if form == "rank":
                y_axis.label.val = "Orthogonal threshold (rank; 1 = lowest)"
            else:
                y_axis.label.val = "Orthogonal threshold (contrast units)"

        elif cond == "lead":
            if form == "rank":
                y_axis.label.val = "Leading context effect (rank; 1 = lowest)"
            else:
                y_axis.label.val = "Leading context effect (contrast units)"

        if form == "linear":
            if cond == "sim":
                y_max = 0.5
            elif cond == "sim_orth":
                y_max = 0.03
                y_axis.TickLabels.format.val = "%.3g"
            elif cond == "lead":
                y_max = 0.085

            y_axis.max.val = y_max
            y_axis.min.val = 0.0

            x_axis.min.val = -2

        else:
            x_axis.min.val = -10
            x_axis.max.val = 105

            y_axis.min.val = -10
            y_axis.max.val = 105

            x_axis.MajorTicks.manualTicks.val = [1] + range(20, 83, 20) + [93]
            x_axis.MinorTicks.hide.val = True

            y_axis.MajorTicks.manualTicks.val = [1] + range(20, 83, 20) + [93]
            y_axis.MinorTicks.hide.val = True

    label = grid.Add("label")

    if save_pdf:
        _save(embed, conf, "ss_timing_{c:s}_{f:s}_scatter".format(c=cond, f=form))

    embed.EnableToolbar(True)
    embed.WaitForClose()



def old_scatter(save_pdf=False, cond="sim"):

    conf = ss_timing_analysis.conf.get_conf()

    # load the fit parameters, excluding the bad subjects
    # this will be subj x onsets x oris x (a,b) x (est, 2.5, 97.5)
    (fit, _, _) = ss_timing_analysis.group_fit.load_fit_data(
        exclude=True
    )

    # restrict to just the alpha estimates
    data = fit[..., 0, 0]

    if cond == "sim":
        data = data[:, 1, 1] - data[:, 1, 0]
    elif cond == "sim_orth":
        data = data[:, 1, 0]
    elif cond == "lead":
        data = data[:, 0, 1] - data[:, 0, 0]

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

    grid.scaleCols.val = [0.85, 0.17]

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

    if cond == "sim":
        y_axis.label.val = "Context effect for simultaneous (par - orth)"
        y_max = 0.5
    elif cond == "sim_orth":
        y_axis.label.val = "Contrast detection threshold for simultaneous, orth"
        y_max = 0.03
        y_axis.TickLabels.format.val = "%.3g"

    elif cond == "lead":
        y_axis.label.val = "Context effect for leading surround (par - orth)"
        y_max = 0.085

    y_axis.max.val = y_max
    y_axis.min.val = 0.0

    # KDE
    graph = grid.Add("graph", autoadd=False)

    graph.leftMargin.val = "0cm"
    graph.bottomMargin.val = "1cm"

    x_axis = graph.Add("axis")
    y_axis = graph.Add("axis")

    kde = scipy.stats.gaussian_kde(data)

    kde_x = np.linspace(0, y_max, 100)
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

    y_axis.max.val = y_max
    x_axis.hide.val = True
    x_axis.lowerPosition.val = 0.075

    if save_pdf:
        _save(embed, conf, "ss_timing_{c:s}_scatter".format(c=cond))

    embed.EnableToolbar(True)
    embed.WaitForClose()


def scatter_sub(save_pdf=False):

    conf = ss_timing_analysis.conf.get_conf()

    # load the fit parameters, excluding the bad subjects
    # this will be subj x onsets x oris x (a,b) x (est, 2.5, 97.5)
    (fit, _, _) = ss_timing_analysis.group_fit.load_fit_data(
        exclude=True
    )

    # restrict to just the alpha estimates
    data = fit[..., 0, 0]

    data = data[:, 1, 1] - data[:, 1, 0]

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")

    page.width.val = "18cm"
    page.height.val = "15cm"

    grid = page.Add("grid")

    grid.rows.val = 2
    grid.columns.val = 2

    grid.leftMargin.val = grid.rightMargin.val = "0cm"
    grid.topMargin.val = "0cm"
    grid.bottomMargin.val = "0.1cm"

    ss_nice = [
        "Unusual experiences",
        "Cognitive disorganisation",
        "Introvertive anhedonia",
        "Impulsive nonconformity"
    ]

    for (i_sub, subscale) in enumerate(
        ("un_ex", "cog_dis", "int_anh", "imp_non")
    ):

        curr_ss = ss_timing_analysis.dem.get_olife_subscale(
            subscale,
            exclude=True
        )

        assert len(curr_ss) == len(data)

        graph = grid.Add("graph", autoadd=False)
        graph.bottomMargin.val = "1cm"
        graph.topMargin.val = "0.6cm"

        label = graph.Add("label")

        label.label.val = ss_nice[i_sub]
        label.yPos.val = 1.01
        label.xPos.val = 0.5
        label.alignHorz.val = "centre"

        x_axis = graph.Add("axis")
        y_axis = graph.Add("axis")

        xy = graph.Add("xy")

        xy.xData.val = curr_ss
        xy.yData.val = data
        xy.PlotLine.hide.val = True
        xy.MarkerFill.transparency.val = 60
        xy.MarkerLine.hide.val = True

        x_axis.label.val = "Score"

        y_axis.label.val = "Context effect for simultaneous (par - orth)"
        y_max = 0.5

        y_axis.max.val = y_max
        y_axis.min.val = 0.0

        x_axis.min.val = -2


    if save_pdf:
        _save(embed, conf, "ss_timing_scatter_sub")

    embed.EnableToolbar(True)
    embed.WaitForClose()


def pairwise_corr(save_pdf=False):

    conf = ss_timing_analysis.conf.get_conf()

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")

    page.width.val = "18cm"
    page.height.val = "15cm"

    grid = page.Add("grid")

    grid.rows.val = 4
    grid.columns.val = 4

    grid.leftMargin.val = grid.rightMargin.val = "0cm"
    grid.topMargin.val = "0cm"
    grid.bottomMargin.val = "0.1cm"

#    y_max = [20, 22, 15, 17]

    ss_nice = [
        "Unusual\\\\experiences",
        "Cognitive\\\\disorganisation",
        "Introvertive\\\\anhedonia",
        "Impulsive\\\\nonconformity"
    ]

    for (i_row, row_sub) in enumerate(
        ("un_ex", "cog_dis", "int_anh", "imp_non")
    ):

        row_ss = ss_timing_analysis.dem.get_olife_subscale(
            row_sub,
            exclude=True
        )

        row_ss = scipy.stats.rankdata(row_ss)

        for (i_col, col_sub) in enumerate(
            ("un_ex", "cog_dis", "int_anh", "imp_non")
        ):

            col_ss = ss_timing_analysis.dem.get_olife_subscale(
                col_sub,
                exclude=True
            )

            col_ss = scipy.stats.rankdata(col_ss)

            graph = grid.Add("graph", autoadd=False)
            graph.leftMargin.val = graph.rightMargin.val = "0.3cm"
            graph.bottomMargin.val = graph.topMargin.val = "0.3cm"
            graph.aspect.val = 1

            x_axis = graph.Add("axis")
            y_axis = graph.Add("axis")

            if i_col < i_row:

                print scipy.stats.pearsonr(row_ss, col_ss)

                xy = graph.Add("xy")

                xy.xData.val = row_ss
                xy.yData.val = col_ss

                xy.PlotLine.hide.val = True
                xy.MarkerFill.transparency.val = 60
                xy.MarkerLine.hide.val = True
                xy.markerSize.val = "2pt"

#                x_axis.label.val = ss_nice[i_row]
#                y_axis.label.val = ss_nice[i_col]

                x_axis.min.val = -10
                x_axis.max.val = 105

                y_axis.min.val = -10
                y_axis.max.val = 105

                x_axis.MajorTicks.manualTicks.val = [1] + range(20, 83, 20) + [93]
                x_axis.MinorTicks.hide.val = True

                y_axis.MajorTicks.manualTicks.val = [1] + range(20, 83, 20) + [93]
                y_axis.MinorTicks.hide.val = True

            else:

                for ax in (x_axis, y_axis):
                    ax.autoMirror.val = True
                    ax.MajorTicks.hide.val = True
                    ax.MinorTicks.hide.val = True
                    ax.TickLabels.hide.val = True

                label = graph.Add("label")
                label.alignHorz.val = "centre"
                label.alignVert.val = "centre"

                if i_row == i_col:
                    label.label.val = ss_nice[i_row]

                else:
                    (r, p) = scipy.stats.pearsonr(row_ss, col_ss)

                    r = np.round(r, 2)

                    r_str = "{r:.02f}".format(r=r)

                    p = np.round(p, 3)

                    if p < 0.001:
                        p_str = "< 0.001"
                    else:
                        p_str = "= {p:.03f}".format(p=p)

                    label.label.val = "\\textit{{r}} = {r:s}\\\\ \\textit{{p}} {p:s}".format(
                        r=r_str, p=p_str
                    )


    if save_pdf:
        _save(embed, conf, "ss_timing_pairwise")

    embed.EnableToolbar(True)
    embed.WaitForClose()



def norms_comparison(save_pdf=False):

    norms = {
        "un_ex": {
            "F": {
                0.25: 4,
                0.5: 9,
                0.75: 15,
                0.9: 19.2
            },
            "M": {
                0.25: 4,
                0.5: 9,
                0.75: 15,
                0.9: 19
            }
        },
        "cog_dis": {
            "F": {
                0.25: 8,
                0.5: 13,
                0.75: 17,
                0.9: 21
            },
            "M": {
                0.25: 8,
                0.5: 12,
                0.75: 16,
                0.9: 20
            }
        },
        "int_anh": {
            "F": {
                0.25: 2,
                0.5: 4,
                0.75: 7,
                0.9: 10
            },
            "M": {
                0.25: 2,
                0.5: 5,
                0.75: 8,
                0.9: 11
            }
        },
        "imp_non": {
            "F": {
                0.25: 6,
                0.5: 9,
                0.75: 12,
                0.9: 14
            },
            "M": {
                0.25: 6,
                0.5: 10,
                0.75: 13,
                0.9: 15
            }
        }
    }

    conf = ss_timing_analysis.conf.get_conf()

    dem = ss_timing_analysis.dem.demographics()

    genders = np.array(
        [dem[subj_id]["gender"] for subj_id in conf.subj_ids]
    )

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")

    page.width.val = "18cm"
    page.height.val = "15cm"

    grid = page.Add("grid")

    grid.rows.val = 2
    grid.columns.val = 2

    grid.leftMargin.val = grid.rightMargin.val = "0cm"
    grid.topMargin.val = "0cm"
    grid.bottomMargin.val = "0.1cm"

    y_max = [20, 22, 15, 17]

    ss_nice = [
        "Unusual experiences",
        "Cognitive disorganisation",
        "Introvertive anhedonia",
        "Impulsive nonconformity"
    ]

    for (i_sub, subscale) in enumerate(
        ("un_ex", "cog_dis", "int_anh", "imp_non")
    ):

        curr_ss = ss_timing_analysis.dem.get_olife_subscale(
            subscale,
            exclude=True
        )

        graph = grid.Add("graph", autoadd=False)
        graph.bottomMargin.val = "1cm"
        graph.topMargin.val = "0.6cm"

        label = graph.Add("label")

        label.label.val = ss_nice[i_sub]
        label.yPos.val = 1.01
        label.xPos.val = 0.5
        label.alignHorz.val = "centre"

        x_axis = graph.Add("axis")
        y_axis = graph.Add("axis")

        for (i_gender, gender) in enumerate(["F", "M"]):

            # first, the norms
            boxplot = graph.Add("boxplot")

            boxplot.calculate.val = False
            boxplot.median.val = norms[subscale][gender][0.5]
            boxplot.boxmin.val = norms[subscale][gender][0.25]
            boxplot.boxmax.val = norms[subscale][gender][0.75]
            boxplot.whiskermax.val = norms[subscale][gender][0.9]
            boxplot.whiskermin.val = norms[subscale][gender][0.25]
            boxplot.mean.val = norms[subscale][gender][0.5]

            boxplot.posn.val = i_gender - 0.15

            boxplot.fillfraction.val = 0.15
            boxplot.markerSize.val = "2pt"

            boxplot.labels.val = "Norm\\\\({g:s})".format(g=gender)
            boxplot.meanmarker.val = "none"

            curr_gender = curr_ss[genders == gender]

            (b25, b50, b75, b90) = scipy.stats.scoreatpercentile(
                curr_gender,
                [25, 50, 75, 90]
            )

            boxplot = graph.Add("boxplot")

            boxplot.calculate.val = False
            boxplot.median.val = b50
            boxplot.boxmin.val = b25
            boxplot.boxmax.val = b75
            boxplot.whiskermax.val = b90
            boxplot.whiskermin.val = b25
            boxplot.mean.val = b50

            boxplot.posn.val = i_gender + 0.15

            boxplot.fillfraction.val = 0.15
            boxplot.markerSize.val = "2pt"

            boxplot.labels.val = "Curr\\\\({g:s})".format(g=gender)

            boxplot.meanmarker.val = "none"

        y_axis.min.val = 0.0
        y_axis.max.val = y_max[i_sub]
        y_axis.MajorTicks.manualTicks.val = range(0, y_max[i_sub] + 1, 5)

        x_axis.mode.val = "labels"
        x_axis.MajorTicks.manualTicks.val = [-0.15, 0.15, 0.85, 1.15]
        x_axis.MinorTicks.hide.val = True
        x_axis.label.val = "Source and gender"

    if save_pdf:
        _save(embed, conf, "ss_timing_norms_comparison")

    embed.EnableToolbar(True)
    embed.WaitForClose()


def context_by_gender(save_pdf=False):

    conf = ss_timing_analysis.conf.get_conf()

    dem = ss_timing_analysis.dem.demographics()

    genders = np.array(
        [dem[subj_id]["gender"] for subj_id in conf.subj_ids]
    )

    # load the fit parameters, excluding the bad subjects
    # this will be subj x onsets x oris x (a,b) x (est, 2.5, 97.5)
    (fit, _, _) = ss_timing_analysis.group_fit.load_fit_data(
        exclude=True
    )

    # restrict to just the alpha estimates
    data = fit[..., 0, 0]

    # and look at the context effect for simultaneous
    data = data[:, 1, 1] - data[:, 1, 0]

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")
    page.width.val = "12cm"
    page.height.val = "8cm"

    graph = page.Add("graph", autoadd=False)
    graph.bottomMargin.val = "1cm"
    graph.topMargin.val = "0.6cm"

    x_axis = graph.Add("axis")
    y_axis = graph.Add("axis")

    for (i_gender, gender) in enumerate(["F", "M"]):

        curr_gender = data[genders == gender]

        boxplot = graph.Add("boxplot")

        dataset_str = "data_{b:d}".format(b=i_gender)

        embed.SetData(
            dataset_str,
            curr_gender
        )

        boxplot.values.val = dataset_str
        boxplot.posn.val = i_gender
        boxplot.labels.val = "{g:s} (n={n:d})".format(
            g=gender,
            n=len(curr_gender)
        )
        boxplot.fillfraction.val = 0.3
        boxplot.markerSize.val = "2pt"

        x_axis.mode.val = "labels"
        x_axis.MajorTicks.manualTicks.val = [0, 1]
        x_axis.MinorTicks.hide.val = True
        x_axis.label.val = "Gender"

        y_axis.TickLabels.format.val = "%.3g"
        y_axis.min.val = 0.0
        y_axis.max.val = 0.5
        y_axis.label.val = "Context effect for simultaneous (par - orth)"

    if save_pdf:
        _save(embed, conf, "ss_timing_context_by_gender")

    embed.EnableToolbar(True)
    embed.WaitForClose()

def context_by_booth(save_pdf=False):

    conf = ss_timing_analysis.conf.get_conf()

    dem = ss_timing_analysis.dem.demographics()

    booths = np.array(
        [dem[subj_id]["testing_booth"] for subj_id in conf.subj_ids]
    )

    potential_booths = np.unique(booths)

    n_booths = len(potential_booths)
    assert n_booths == 2

    # load the fit parameters, excluding the bad subjects
    # this will be subj x onsets x oris x (a,b) x (est, 2.5, 97.5)
    (fit, _, _) = ss_timing_analysis.group_fit.load_fit_data(
        exclude=True
    )

    # restrict to just the alpha estimates
    data = fit[..., 0, 0]

    # and look at the context effect for simultaneous
    data = data[:, 1, 1] - data[:, 1, 0]

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")
    page.width.val = "12cm"
    page.height.val = "8cm"

    graph = page.Add("graph", autoadd=False)
    graph.bottomMargin.val = "1cm"
    graph.topMargin.val = "0.6cm"

    x_axis = graph.Add("axis")
    y_axis = graph.Add("axis")

    for i_booth in xrange(n_booths):

        curr_booth = data[booths == potential_booths[i_booth]]

        boxplot = graph.Add("boxplot")

        dataset_str = "data_{b:d}".format(b=i_booth)

        embed.SetData(
            dataset_str,
            curr_booth
        )

        boxplot.values.val = dataset_str
        boxplot.posn.val = i_booth
        boxplot.labels.val = "{b:d} (n={n:d})".format(
            b=i_booth + 1,
            n=len(curr_booth)
        )
        boxplot.fillfraction.val = 0.3
        boxplot.markerSize.val = "2pt"

        x_axis.mode.val = "labels"
        x_axis.MajorTicks.manualTicks.val = [0, 1]
        x_axis.MinorTicks.hide.val = True
        x_axis.label.val = "Testing booth"

        y_axis.TickLabels.format.val = "%.3g"
        y_axis.min.val = 0.0
        y_axis.max.val = 0.5
        y_axis.label.val = "Context effect for simultaneous (par - orth)"

    if save_pdf:
        _save(embed, conf, "ss_timing_context_by_booth")

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
        y_axis.label.val = "Threshold contrast"

        cond_label = graph.Add("label")

        cond_label.label.val = conf.surr_onset_labels[i_onset]
        cond_label.yPos.val = 1.02
        cond_label.xPos.val = 0.5
        cond_label.alignHorz.val = "centre"
        cond_label.Text.size.val = "8pt"

    if save_pdf:
        _save(embed, conf, "ss_timing_thresholds")

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
            cond_label.Text.size.val = "8pt"

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
        _save(embed, conf, "ss_timing_eg_subject")

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

        page.width.val = "21cm"
        page.height.val = "29.7cm"

        label = page.Add("label")

        label.label.val = subj_id
        label.yPos.val = 0.77

        if subj_id in conf.exclude_ids:
            label.label.val += " (excluded)"
            label.Text.color.val = "red"

        grid = page.Add("grid")

        grid.rows.val = 2
        grid.columns.val = 2

        grid.leftMargin.val = grid.rightMargin.val = "3cm"
        grid.topMargin.val = "8cm"
        grid.bottomMargin.val = "8cm"

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
                cond_label.Text.size.val = "8pt"

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
        _save(
            embed,
            conf,
            "ss_timing_subjects",
            page=range(conf.n_all_subj)
        )

    embed.EnableToolbar(True)
    embed.WaitForClose()

