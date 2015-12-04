
import os

import numpy as np
import scipy.stats

import xlrd
import savReaderWriter

import veusz.embed
import figutils

import ss_timing_analysis.conf


def load_data():

    conf = ss_timing_analysis.conf.get_conf()

    data_path = os.path.join(
        conf.base_path,
        "s_p_data",
        "Data_Serrano-PedrazaEA2014.xlsx"
    )

    ws = xlrd.open_workbook(data_path).sheets()[0]

    h = "Contrast detection thresholds. Serrano-Pedraza et al. (2014) Frontier in Psychology (5)"

    # check the header is correct
    assert ws.cell(0, 0).value == h

    n_vals = 4

    n_hc = 24
    hc_data = np.empty((n_hc, n_vals))
    hc_data.fill(np.NAN)
    i_row_start_hc = 3
    hc_ages = []

    n_sz = 21
    sz_data = np.empty((n_sz, n_vals))
    sz_data.fill(np.NAN)
    i_row_start_sz = 30
    sz_ages = []

    i_cols = [
        ("ns", 2),
        ("os", 3),
        ("ps", 4)
    ]

    i_age_col = 1

    for i_hc in xrange(n_hc):

        i_subj_row = i_row_start_hc + i_hc

        subj_id = ws.cell(i_subj_row, 0).value

        # make sure the subject ID is as expected
        assert subj_id == "C{n:02d}".format(n=i_hc + 1)

        for (i_cond, (_, i_col)) in enumerate(i_cols):

            datum = float(ws.cell(i_subj_row, i_col).value)

            hc_data[i_hc, i_cond] = datum

        hc_data[i_hc, -1] = float(ws.cell(i_subj_row, i_age_col).value)

    for i_sz in xrange(n_sz):

        i_subj_row = i_row_start_sz + i_sz

        subj_id = ws.cell(i_subj_row, 0).value

        # make sure the subject ID is as expected
        assert subj_id == "SZ{n:02d}".format(n=i_sz + 1)

        for (i_cond, (_, i_col)) in enumerate(i_cols):

            datum = float(ws.cell(i_subj_row, i_col).value)

            sz_data[i_sz, i_cond] = datum

        sz_data[i_sz, -1] = float(ws.cell(i_subj_row, i_age_col).value)

    assert np.sum(np.isnan(hc_data)) == 0
    assert np.sum(np.isnan(sz_data)) == 0

    return (hc_data, sz_data)


def spss_export():

    conf = ss_timing_analysis.conf.get_conf()

    (hc_data, sz_data) = load_data()

    sav_path = os.path.join(
        conf.base_path,
        "s_p_data",
        "s_p_data_for_spss.sav"
    )

    var_names = ["GR", "NS", "OS", "PS"]
    var_types = {
        "GR": 2,
        "NS": 0,
        "OS": 0,
        "PS": 0
    }

    with savReaderWriter.SavWriter(sav_path, var_names, var_types) as writer:

        for i_hc in xrange(hc_data.shape[0]):

            row = ["HC"] + list(hc_data[i_hc, :])

            writer.writerow(row)

        for i_sz in xrange(sz_data.shape[0]):

            row = ["SZ"] + list(sz_data[i_sz, :])

            writer.writerow(row)


def figure_3():

    conf = ss_timing_analysis.conf.get_conf()

    data = load_data()

    (hc_data, sz_data) = data

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")
    page.width.val = "14cm"
    page.height.val = "10cm"

    grid = page.Add("grid")

    grid.rows.val = 1
    grid.columns.val = 3

    y_mins = [0, 0, 0]
    y_maxes = [0.1, 0.1, 0.5]

    cols = ["red", "blue"]
    symbols = ["circle", "square"]

    for i_cond in xrange(3):

        graph = grid.Add("graph", autoadd=False)
        graph.aspect.val = 1

        graph.leftMargin.val = graph.rightMargin.val = "0.5cm"
        graph.topMargin.val = graph.bottomMargin.val = "0cm"

        x_axis = graph.Add("axis")
        y_axis = graph.Add("axis")

        for i_group in xrange(2):

            xy = graph.Add("xy")

            # age
            xy.xData.val = data[i_group][:, -1]
            # cond
            xy.yData.val = data[i_group][:, i_cond]

            xy.PlotLine.hide.val = True
            xy.markerSize.val = "2pt"
            xy.MarkerFill.hide.val = True
            xy.MarkerLine.color.val = cols[i_group]
            xy.marker.val = symbols[i_group]

        y_axis.min.val = y_mins[i_cond]
        y_axis.max.val = y_maxes[i_cond]
        x_axis.min.val = 15
        x_axis.max.val = 65

    fig_path = os.path.join(
        conf.base_path,
        "s_p_data",
        "s_p_fig3.vsz"
    )

    embed.Save(fig_path)

    embed.EnableToolbar(True)
    embed.WaitForClose()


def figure_4():

    conf = ss_timing_analysis.conf.get_conf()

    data = load_data()

    (hc_data, sz_data) = data

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")
    page.width.val = "14cm"
    page.height.val = "10cm"

    grid = page.Add("grid")

    grid.rows.val = 1
    grid.columns.val = 2

    y_mins = [0, -1]
    y_maxes = [2.5, 1]

    cols = ["red", "blue"]
    symbols = ["circle", "square"]

    for (i_panel, i_cond) in enumerate([2, 1]):

        graph = grid.Add("graph", autoadd=False)
        graph.aspect.val = 1

        graph.leftMargin.val = graph.rightMargin.val = "0.5cm"
        graph.topMargin.val = graph.bottomMargin.val = "0cm"

        x_axis = graph.Add("axis")
        y_axis = graph.Add("axis")

        for i_group in xrange(2):

            xy = graph.Add("xy")

            # age
            xy.xData.val = data[i_group][:, -1]

            # no surround
            ns = data[i_group][:, 0]
            # surround; OS or PS
            s = data[i_group][:, i_cond]

            ratio = s / ns
            log_ratio = np.log10(ratio)

            # cond
            xy.yData.val = log_ratio

            xy.PlotLine.hide.val = True
            xy.markerSize.val = "2pt"
            xy.MarkerFill.hide.val = True
            xy.MarkerLine.color.val = cols[i_group]
            xy.marker.val = symbols[i_group]

        y_axis.min.val = y_mins[i_panel]
        y_axis.max.val = y_maxes[i_panel]
        x_axis.min.val = 15
        x_axis.max.val = 65

    fig_path = os.path.join(
        conf.base_path,
        "s_p_data",
        "s_p_fig4.vsz"
    )

    embed.Save(fig_path)

    embed.EnableToolbar(True)
    embed.WaitForClose()

    return log_ratio


def ratio_check():

    conf = ss_timing_analysis.conf.get_conf()

    data = load_data()

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")
    page.width.val = "14cm"
    page.height.val = "20cm"

    grid = page.Add("grid")
    grid.rows.val = 4
    grid.columns.val = 2

    cols = ["red", "blue"]
    symbols = ["circle", "square"]

    for i_group in xrange(2):

        curr_data = data[i_group]

        ns = curr_data[:, 0]

        for i_var in [1, 2]:

            sc = curr_data[:, i_var]

            for graph_type in ["NSxS", "NSx(S/NS)"]:

                graph = grid.Add("graph", autoadd=False)
                graph.aspect.val = 1

                x_axis = graph.Add("axis")
                y_axis = graph.Add("axis")

                x_axis.autoRange.val = "+10%"
                y_axis.autoRange.val = "+10%"

                xy = graph.Add("xy")

                xy.xData.val = ns

                if graph_type == "NSxS":
                    xy.yData.val = sc

                    b = np.std(sc, ddof=1) / np.std(ns, ddof=1)

                    a = np.mean(sc) - b * np.mean(ns)

                    x_max = np.max(ns)
                    y_max = np.max(sc)

                    l_x = np.linspace(0, x_max, 100)
                    l_y = l_x * b + a

                    f = graph.Add("xy")

                    f.xData.val = l_x
                    f.yData.val = l_y
                    f.PlotLine.hide.val = False
                    f.markerSize.val = "0pt"
                    f.MarkerFill.hide.val = True
                    f.MarkerLine.color.val = cols[i_group]
                    f.marker.val = symbols[i_group]

                elif graph_type == "NSx(S/NS)":
                    xy.yData.val = np.log10(sc / ns)

                    print scipy.stats.spearmanr(ns, np.log10(sc / ns))

                xy.PlotLine.hide.val = True
                xy.markerSize.val = "2pt"
                xy.MarkerFill.hide.val = True
                xy.MarkerLine.color.val = cols[i_group]
                xy.marker.val = symbols[i_group]

                x_axis.min.val = 0
                #y_axis.min.val = 0

    embed.EnableToolbar(True)
    embed.WaitForClose()







