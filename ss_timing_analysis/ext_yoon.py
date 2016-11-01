import os
import collections

import numpy as np
import scipy.stats

import pyvttbl

import veusz.embed

import figutils

import ss_timing_analysis.conf


def load_data():

    conf = ss_timing_analysis.conf.get_conf()

    data_path = os.path.join(
        conf.base_path,
        "yoon2009-schizophrenia-bulletin",
        "Yoon2009data.csv"
    )

    data_dict = collections.defaultdict(dict)

    with open(data_path, "r") as data_file:

        header = data_file.readline().strip().split(",")

        data_str = data_file.readlines()

        for row_str in data_str:

            row = row_str.strip().split(",")

            subj_id = row[header.index("SubjectID")]

            group = row[header.index("Group")]

            surr = row[header.index("Surround")]

            threshold = float(row[header.index("Threshold")])

            if "group" in data_dict[subj_id].keys():
                assert group == data_dict[subj_id]["group"]

            data_dict[subj_id]["group"] = group
            data_dict[subj_id][surr] = threshold

    n_control = 0
    n_patient = 0

    for subj_info in data_dict.values():

        group = subj_info["group"]

        if group == "C":
            n_control += 1
        elif group == "P":
            n_patient += 1
        else:
            raise ValueError("Unknown group")

    controls = np.empty((n_control, 3))
    controls.fill(np.nan)

    i_control = 0

    patients = np.empty((n_patient, 3))
    patients.fill(np.nan)

    i_patient = 0

    for subj_info in data_dict.values():

        thresh_n = subj_info["None"]
        thresh_p = subj_info["Para"]
        thresh_o = subj_info["Ortho"]

        group = subj_info["group"]

        if group == "C":
            controls[i_control, :] = (thresh_n, thresh_o, thresh_p)
            i_control += 1

        elif group == "P":
            patients[i_patient, :] = (thresh_n, thresh_o, thresh_p)
            i_patient += 1

    assert i_control == n_control
    assert i_patient == n_patient
    assert np.sum(np.isnan(controls)) == 0
    assert np.sum(np.isnan(patients)) == 0

    return (controls, patients)


def check():

    data = load_data()
    ratios = get_ratios()

    for (src_str, src, src_r) in zip(("Control", "Patient"), data, ratios):

        print src_str + ":"

        for s in (src, src_r):

            print "\t{m:s}".format(m=", ".join(map(str, np.mean(s, axis=0))))


def ratio_anova():

    ratios = get_ratios()

    data_dict = {
        "SUBJECT": [],
        "GROUP": [],
        "SURR": [],
        "RATIO": []
    }

    subj_num =  1

    for (grp_data, grp) in zip(ratios, ("C", "P")):

        for i_subj in xrange(grp_data.shape[0]):

            for (i_s, sc) in enumerate(("O", "P")):

                data_dict["SUBJECT"].append(subj_num)
                data_dict["GROUP"].append(grp)
                data_dict["SURR"].append(sc)
                data_dict["RATIO"].append(grp_data[i_subj, i_s])

            subj_num += 1

    df = pyvttbl.DataFrame(data_dict)

    anova = df.anova(
        dv="RATIO",
        wfactors=["SURR"],
        bfactors=["GROUP"]
    )

    for factors in (("GROUP",), ("SURR", ), ("SURR", "GROUP")):

        af = anova[factors]

        print factors
        print "\tF({a:.0f},{b:.0f}) = {f:.4f}, p = {p:.4f}".format(
            a=af["df"], b=af["dfe"], f=af["F"], p=af["p"]
        )

    return anova


def write_ratios_for_spss():

    conf = ss_timing_analysis.conf.get_conf()

    ratios = get_ratios()

    data_path = os.path.join(conf.base_path, "ss_timing_yoon_spss.tsv")

    with open(data_path, "w") as data_file:

        for (grp_data, grp) in zip(ratios, ("C", "P")):

            for i_subj in xrange(grp_data.shape[0]):

                row = [grp]

                for i_s in xrange(2):

                    row.append(str(grp_data[i_subj, i_s]))

                data_file.write("\t".join(row) + "\n")


def get_ratios():

    (control_data, patient_data) = load_data()

    control_ratios = np.empty((control_data.shape[0], 2))
    control_ratios.fill(np.nan)

    patient_ratios = np.empty((patient_data.shape[0], 2))
    patient_ratios.fill(np.nan)

    for (in_data, out_data) in zip(
        (control_data, patient_data),
        (control_ratios, patient_ratios)
    ):

        # orth / none
        out_data[:, 0] = in_data[:, 1] / in_data[:, 0]

        # par / none
        out_data[:, 1] = in_data[:, 2] / in_data[:, 0]

    return (control_ratios, patient_ratios)


def regress(x, y, n_boot=10000):

    coef = np.empty((2, n_boot + 1))
    coef.fill(np.nan)

    n_subj = len(x)

    i_subj = np.arange(n_subj)

    for i_boot in xrange(n_boot + 1):

        # 'type II regression', following Ludbrook (2011), since there is error in
        # both x and y

        b = np.std(y[i_subj]) / np.std(x[i_subj])

        a = np.mean(y[i_subj]) - b * np.mean(x[i_subj])

        coef[:, i_boot] = (a, b)

        i_subj = np.random.choice(n_subj, n_subj, replace=True)

    return np.squeeze(coef)


def regress_ci(coef, n_x=100):

    n_boot = coef.shape[1] - 1

    x = np.linspace(0, 100, n_x)

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


def figure():

    conf = ss_timing_analysis.conf.get_conf()

    data = load_data()
    ratios = get_ratios()

    embed = veusz.embed.Embedded("veusz")
    figutils.set_veusz_style(embed)

    page = embed.Root.Add("page")

    page.width.val = "17cm"
    page.height.val = "20cm"

    grid = page.Add("grid")

    grid.columns.val = 2
    grid.rows.val = 4

    fine_x = np.linspace(0, 100, 100)

    for (curr_data, curr_ratios, curr_group) in zip(data, ratios, ("C", "P")):

        i_denom = 0

        for i_num in (1, 2):

            graph = grid.Add("graph", autoadd=False)

            graph.aspect.val = 1.0

            x_axis = graph.Add("axis")
            y_axis = graph.Add("axis")

            points = graph.Add("xy")

            points.xData.val = curr_data[:, i_denom]
            points.yData.val = curr_data[:, i_num]

            points.PlotLine.hide.val = True
            points.MarkerLine.hide.val = True
            points.MarkerFill.transparency.val = 80
            points.markerSize.val = "2pt"

            coefs = regress(curr_data[:, i_denom], curr_data[:, i_num])

            fit = regress_ci(coefs)

            fit_str = "fit_{g:s}_{n:d}".format(g=curr_group, n=i_num)

            embed.SetData(
                fit_str,
                fit[0, :],
                poserr=abs(fit[2, :] - fit[0, :]),
                negerr=abs(fit[0, :] - fit[1, :])
            )

            fit_xy = graph.Add("xy")

            fit_xy.xData.val = fine_x
            fit_xy.yData.val = fit_str

            fit_xy.MarkerFill.hide.val = fit_xy.MarkerLine.hide.val = True
            fit_xy.errorStyle.val = "linevert"
            fit_xy.ErrorBarLine.style.val = "dashed"

            x_axis.min.val = y_axis.min.val = 0.0
            x_axis.max.val = y_axis.max.val = 100.0

            x_axis.outerticks.val = y_axis.outerticks.val = True

            r_graph = grid.Add("graph", autoadd=False)

            r_graph.aspect.val = 1.0

            x_axis = r_graph.Add("axis")
            y_axis = r_graph.Add("axis")

            r_xy = r_graph.Add("xy")

            r_xy.xData.val = curr_data[:, i_denom]
            r_xy.yData.val = curr_ratios[:, (i_num - 1)]

            r_xy.PlotLine.hide.val = True

            r_fit = fit[0, 1:] / fine_x[1:]

            fit_r = r_graph.Add("xy")

            fit_r.xData.val = fine_x[1:]
            fit_r.yData.val = r_fit

            fit_r.MarkerFill.hide.val = fit_r.MarkerLine.hide.val = True

    stem = os.path.join(
        conf.figures_path,
        "ss_timing_yoon"
    )

    embed.Save(stem + ".vsz")

    embed.WaitForClose()


def perm_regress_test():

    (c_data, p_data) = load_data()

    n_c = c_data.shape[0]
    n_p = p_data.shape[0]

    b_data = np.vstack((c_data, p_data))

    boot_diff = np.empty((10000, 2))
    boot_diff.fill(np.nan)

    for i_boot in xrange(10000):

        for i_ori in [1, 2]:

            i = np.random.permutation(n_c + n_p)

            i_c = i[:n_c]
            i_p = i[n_c:]

            assert len(i_p) == n_p

            c_data = b_data[i_c, :]

            cx = c_data[:, 0]
            cy = c_data[:, i_ori]

            c_coef = regress(cx, cy, n_boot=0)[1]
            c_coef = cy / cx

            p_data = b_data[i_p, :]

            px = p_data[:, 0]
            py = p_data[:, i_ori]

            p_coef = regress(px, py, n_boot=0)[1]
            p_coef = py / px

            boot_diff[i_boot, i_ori - 1] = np.mean(c_coef) - np.mean(p_coef)

    return boot_diff


def bootstrap_regress_test():

    (c_data, p_data) = load_data()

    n_c = c_data.shape[0]
    n_p = p_data.shape[0]

    boot_diff = np.empty((10000, 2))
    boot_diff.fill(np.nan)

    for i_boot in xrange(10000):

        for i_ori in [1, 2]:

            i_c = np.random.choice(n_c, n_c, replace=True)
            i_p = np.random.choice(n_p, n_p, replace=True)

            cx = c_data[i_c, 0]
            cy = c_data[i_c, i_ori]

            c_coef = regress(cx, cy, n_boot=0)[1]

            px = p_data[i_p, 0]
            py = p_data[i_p, i_ori]

            p_coef = regress(px, py, n_boot=0)[1]

            boot_diff[i_boot, i_ori - 1] = c_coef - p_coef

    return boot_diff


def bootstrap_diff_test():

    (c_data, p_data) = load_data()

    (rc_data, rp_data) = get_ratios()

    n_c = c_data.shape[0]
    n_p = p_data.shape[0]

    boot_diffs = np.empty((10000, 2))
    boot_diffs.fill(np.nan)

    for i_boot in xrange(10000):

        for i_ori in xrange(2):

            i_c = np.random.choice(n_c, n_c, replace=True)
            i_p = np.random.choice(n_p, n_p, replace=True)

            d = np.mean(rc_data[i_c, i_ori]) - np.mean(rp_data[i_p, i_ori])

            boot_diffs[i_boot, i_ori] = d

    return boot_diffs
