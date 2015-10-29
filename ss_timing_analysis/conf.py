import os
import collections

import numpy as np

import xlrd

import ss_timing.conf


def get_conf(subj_id=""):

    conf = ss_timing.conf.get_conf(subj_id)

    conf.demographics_path = "/home/damien/venv_study/ss_timing/demographics"
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
        "p1079",
        "p1098"
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
