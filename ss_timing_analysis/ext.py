import os

import numpy as np
import scipy.stats

import veusz.embed

import figutils

import ss_timing_analysis.conf
import ss_timing_analysis.ext_yoon
import ss_timing_analysis.ext_s_p


def figure(study):

    conf = ss_timing_analysis.conf.get_conf()

    if study == "yoon":
        study_pkg = ss_timing_analysis.ext_yoon
    elif study == "s-p":
        study_pkg = ss_timing_analysis.ext_s_p

    # two-item list (controls, patients)
    # each item is an array, n subj x 3 (ns, os, ps)
    data = study_pkg.load_data()

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    # x to evaluate for the regressions
    fine_x = np.linspace(0, 100, 101)

    if study == "s-p":
        fine_x /= 100.0

    nice_surrs = {"O": "Orthogonal", "P": "Parallel"}
    nice_grps = {"C": "Control", "P": "Patient"}

    limits = {
        "yoon": {
            "C": {
                "x": 40.0,
                "y": {"O": 100.0, "P": 100.0},
                "r": {"O": [0.5, 2.25], "P": [1.4, 4.0]}
            },
            "P": {
                "x": 40.0,
                "y": {"O": 100.0, "P": 100.0},
                "r": {"O": [0.5, 4.0], "P": [1.25, 3.75]}
            }
        },
        "s-p": {
            "C": {
                "x": 0.04,
                "y": {"O": 0.03, "P": 0.6},
                "r": {"O": [-1.2, 1.1], "P": [0.5, 4.25]}
            },
            "P": {
                "x": 0.1,
                "y": {"O": 0.06, "P": 0.5},
                "r": {"O": [-2.2, 0.5], "P": [0, 3.05]}
            }
        }
    }


    study_limits = limits[study]

    page = embed.Root.Add("page")

    page.width.val = "12.5cm"
    page.height.val = "20cm"

    grp_grid = page.Add("grid")

    grp_grid.rows.val = 2
    grp_grid.columns.val = 1

    grp_grid.leftMargin.val = grp_grid.rightMargin.val = "0cm"
    grp_grid.topMargin.val = grp_grid.bottomMargin.val = "0cm"

    for (i_grp, (grp_data, grp_name)) in enumerate(zip(data, ("C", "P"))):

        grid = grp_grid.Add("grid")

        # OS, PS
        grid.rows.val = 2
        # NS x AS, (AS/NS) x NS
        grid.columns.val = 2

        grid.leftMargin.val = grid.rightMargin.val = grid.bottomMargin.val = "0cm"
        grid.topMargin.val = "1cm"

        label = page.Add("label")

        label.label.val = nice_grps[grp_name]
        label.Text.bold.val = True
        label.Text.size.val = "10pt"
        label.alignHorz.val = "centre"
        label.alignVert.val = "centre"

        label.yPos.val = 1 - (0.5 * i_grp + 0.025)

        # denominator is the no-surround condition
        denom = grp_data[:, 0]

        # loop over the orthogonal and parallel conditions
        for (i_num, num_name) in enumerate(("O", "P"), 1):

            numerator = grp_data[:, i_num]

            # first, NS x AS graph
            graph = grid.Add("graph", autoadd=False)

            x_axis = graph.Add("axis")
            y_axis = graph.Add("axis")

            xy = graph.Add("xy")

            xy.xData.val = denom
            xy.yData.val = numerator

            _format_points(xy)

            coef = regress(x=denom, y=numerator)

            fit_y = regress_ci(coef=coef, x=fine_x)

            fit_y_str = "_".join([grp_name, num_name])

            embed.SetData(
                fit_y_str,
                fit_y[0, :],
                poserr=abs(fit_y[2, :] - fit_y[0, :]),
                negerr=abs(fit_y[0, :] - fit_y[1, :])
            )

            fit_xy = graph.Add("xy")

            fit_xy.xData.val = fine_x
            fit_xy.yData.val = fit_y_str

            fit_xy.MarkerFill.hide.val = fit_xy.MarkerLine.hide.val = True
            fit_xy.errorStyle.val = "linevert"
            fit_xy.ErrorBarLine.style.val = "dashed"

            x_axis.label.val = "No-surround threshold"
            y_axis.label.val = "{s:s} threshold".format(
                s=nice_surrs[num_name]
            )

            x_axis.min.val = 0.0
            x_axis.max.val = study_limits[grp_name]["x"]

            y_axis.min.val = 0.0
            y_axis.max.val = study_limits[grp_name]["y"][num_name]

            x_axis.outerticks.val = y_axis.outerticks.val = True

            # now, the ratio
            graph = grid.Add("graph", autoadd=False)

            x_axis = graph.Add("axis")
            y_axis = graph.Add("axis")

            x_axis.outerticks.val = y_axis.outerticks.val = True

            ratio = numerator / denom

            # take the log ratio if the study is Serrano-Pedraza et al.
            if study == "s-p":
                ratio = np.log(ratio)

            xy = graph.Add("xy")

            xy.xData.val = denom
            xy.yData.val = ratio

            _format_points(xy)

            # calculate the correlation
            (r, p) = scipy.stats.spearmanr(denom, ratio)

            c_str = "\italic{{r}}_{{s}} = {r:.2f}\\\\\italic{{p}} = {p:.3f}".format(r=r, p=p)

            c_label = graph.Add("label")

            c_label.label.val = c_str

            c_label.xPos.val = 0.75
            c_label.yPos.val = 0.8
            c_label.Text.size.val = "6pt"

            x_axis.label.val = "No-surround threshold"
            y_axis.label.val = "{s:s} suppression index".format(
                s=nice_surrs[num_name]
            )

            x_axis.min.val = 0.0
            x_axis.max.val = study_limits[grp_name]["x"]

            (y_axis.min.val, y_axis.max.val) = study_limits[grp_name]["r"][num_name]


    stem = os.path.join(
        conf.figures_path,
        "_".join(("ss_timing", study))
    )

    #embed.Zoom(0.5)
    embed.Save(stem + ".vsz")
    embed.Export(stem + ".pdf")

    embed.WaitForClose()


def _format_points(xy):

    xy.PlotLine.hide.val = True
    xy.MarkerLine.hide.val = True
    xy.markerSize.val = "2pt"


def regress(x, y, n_boot=10000):

    coef = np.empty((2, n_boot + 1))
    coef.fill(np.nan)

    n_subj = len(x)

    i_subj = np.arange(n_subj)

    for i_boot in xrange(n_boot + 1):

        # 'type II regression', following Ludbrook (2011), since there is error in
        # both x and y

        b = np.std(y[i_subj], ddof=1) / np.std(x[i_subj], ddof=1)

        a = np.mean(y[i_subj]) - b * np.mean(x[i_subj])

        coef[:, i_boot] = (a, b)

        i_subj = np.random.choice(n_subj, n_subj, replace=True)

    return np.squeeze(coef)


def regress_ci(coef, x=None):

    n_boot = coef.shape[1] - 1

    if x is None:
        x = np.linspace(0, 100, 101)

    n_x = len(x)

    y = np.empty((3, n_x))
    y.fill(np.nan)

    y[0, :] = coef[1, 0] * x + coef[0, 0]

    boot_y = np.empty((n_boot, n_x))
    boot_y.fill(np.nan)

    for i_boot in xrange(n_boot):
        boot_y[i_boot, :] = coef[1, i_boot + 1] * x + coef[0, i_boot + 1]

    y[1:, :] = scipy.stats.scoreatpercentile(boot_y, [2.5, 97.5], axis=0)

    assert np.sum(np.isnan(y)) == 0

    return y
