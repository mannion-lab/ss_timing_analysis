import os

import numpy as np

import xlrd

import ss_timing.conf


def get_conf(subj_id=""):

    conf = ss_timing.conf.get_conf(subj_id)

    conf.demographics_path = "/home/damien/venv_study/ss_timing/demographics"
    conf.demographics_date = "20151019"

    conf.group_data_path = "/home/damien/venv_study/ss_timing/group_data"
    conf.figures_path = "/home/damien/venv_study/ss_timing/figures"

    conf.surr_ori_labels = ["Orthogonal", "Parallel"]
    conf.surr_onset_labels = ["Leading surround", "Simultaneous"]

    conf.exclude_ids = [
        "p1027",
        "p1039",
        "p1048",
        "p1051",
        "p1060",
        "p1079"
    ]

    conf.missing_ids = [
        1040  # started but wrong experiment
    ]

    min_id = 1001
    max_id = 1101

    conf.all_subj_ids = [
        "p{n:d}".format(n=n)
        for n in range(min_id, max_id + 1)
        if n not in conf.missing_ids
    ]

    conf.subj_ids = [
        subj_id
        for subj_id in conf.all_subj_ids
        if subj_id not in conf.exclude_ids
    ]

    conf.n_all_subj = len(conf.all_subj_ids)
    conf.n_subj = len(conf.subj_ids)

    conf.demographics = demographics(conf)

    conf.log_bin_size = 0.03

    conf.log_bin_edges = np.arange(
        np.log10(0.001) - conf.log_bin_size / 2.0,
        np.log10(1.0) + conf.log_bin_size / 2.0,
        conf.log_bin_size
    )
    conf.bin_edges = 10 ** conf.log_bin_edges

    conf.n_bins = len(conf.bin_edges)

    conf.log_bin_centres = conf.log_bin_edges - conf.log_bin_size / 2.0
    conf.bin_centres = 10 ** conf.log_bin_centres

    conf.fine_x = np.logspace(np.log10(0.001), np.log10(1.0), 100)
    conf.n_fine_x = len(conf.fine_x)

    conf.n_boot = 10000
    conf.boot_seed = 2118217324

    return conf


def demographics(conf):

    dem = {}

    xls_path = os.path.join(
        conf.demographics_path,
        "sstimingspreadsheet_" + conf.demographics_date + ".xlsx"
    )

    wb = xlrd.open_workbook(xls_path).sheets()[0]

    keys = [wb.cell(0, i_col).value for i_col in xrange(wb.ncols)]

    for i_row in range(1, wb.nrows):

        subj_id = (
            "p{n:d}".format(
                n=int(wb.cell(i_row, keys.index("Participant ID")).value)
            )
        )

        dem[subj_id] = {
            key: wb.cell(i_row, keys.index(key)).value
            for key in keys
            if key != "Participant ID"
        }

        dem[subj_id]["olt"] = (
            dem[subj_id]["CogDis"] +
            dem[subj_id]["ImpNon"] +
            dem[subj_id]["IntAnh"] +
            dem[subj_id]["UnEx"]
        )

    return dem
