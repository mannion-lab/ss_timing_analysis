import os
import collections

import numpy as np
import scipy.stats

import pyvttbl

import veusz.embed

import figutils

import ss_timing_analysis.conf
import ss_timing_analysis.ext


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

            # as per the paper, but applied already but not to all?
            if threshold > 75.0:
                threshold = 75.0

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
    ratios = ss_timing_analysis.ext.get_ratios(data, log_xform=False)

    for (src_str, src, src_r) in zip(("Control", "Patient"), data, ratios):

        print src_str + ":"

        for s in (src, src_r):

            print "\t{m:s}".format(m=", ".join(map(str, np.mean(s, axis=0))))


