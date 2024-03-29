import os
import csv

import numpy as np

import ss_timing_analysis.conf
import ss_timing_analysis.data
import ss_timing_analysis.dem


def save_group_data():

    conf = ss_timing_analysis.conf.get_conf()

    data = np.empty(
        (
            conf.n_all_subj,  # this is prior to any exclusions
            conf.n_surr_onsets,
            conf.n_surr_oris,
            (
                conf.n_trials_per_stair *
                conf.n_stairs_per_run *
                conf.n_runs_per_cond
            ),
            2
        )
    )
    data.fill(np.NAN)

    for (i_subj, subj_id) in enumerate(conf.all_subj_ids):
        data[i_subj, ...] = ss_timing_analysis.data.get_subj_resp_data(subj_id)

    assert np.sum(np.isnan(data)) == 0

    npy_path = os.path.join(
        conf.group_data_path,
        "ss_timing_all_subj_resp_data.npy"
    )

    np.save(npy_path, data)

    return data


def load_group_data(exclude=False):

    conf = ss_timing_analysis.conf.get_conf()

    npy_path = os.path.join(
        conf.group_data_path,
        "ss_timing_all_subj_resp_data.npy"
    )

    data = np.load(npy_path)

    if exclude:

        # work out which indices correspond to the bad subjects
        i_bad = [
            conf.all_subj_ids.index(bad_subj_id)
            for bad_subj_id in conf.exclude_ids
        ]

        # and chop those indices out
        data = np.delete(data, i_bad, axis=0)

    return data


def save_group_data_2d():

    conf = ss_timing_analysis.conf.get_conf()

    data = load_group_data(exclude=False)

    n_trials_per_cond = data.shape[-2]

    data_rows = []

    for i_subj in xrange(conf.n_all_subj):
        for i_onset in xrange(conf.n_surr_onsets):
            for i_ori in xrange(conf.n_surr_oris):
                for i_trial in xrange(n_trials_per_cond):

                    (contrast, resp) = data[
                        i_subj,
                        i_onset,
                        i_ori,
                        i_trial,
                        :
                    ]

                    row = [
                        i_subj,
                        i_onset,
                        i_ori,
                        i_trial,
                        contrast,
                        resp
                    ]

                    data_rows.append(row)

    data_rows = np.array(data_rows)

    npy_path = os.path.join(
        conf.group_data_path,
        "ss_timing_all_subj_resp_data_2d.npy"
    )

    np.save(npy_path, data_rows)

    return data_rows


def load_group_data_2d(exclude=False):

    conf = ss_timing_analysis.conf.get_conf()

    npy_path = os.path.join(
        conf.group_data_path,
        "ss_timing_all_subj_resp_data_2d.npy"
    )

    data = np.load(npy_path)

    if exclude:

        i_bad = [
            conf.all_subj_ids.index(bad_subj_id)
            for bad_subj_id in conf.exclude_ids
        ]

        i_bad_rows = [
            data[:, 0] == i_bad_subj_id
            for i_bad_subj_id in i_bad
        ]

        i_bad_rows = np.where(np.any(i_bad_rows, axis=0))[0]

        # remove
        data = np.delete(data, i_bad_rows, axis=0)

        # but now we want to add a column with the linear subject indices
        n_trials_per_subj = (
            conf.n_surr_onsets *
            conf.n_surr_oris *
            conf.n_trials_per_stair *
            conf.n_stairs_per_run *
            conf.n_runs_per_cond
        )

        new_i_subj = np.repeat(range(conf.n_subj), n_trials_per_subj)

        # add in our new column after the previous column
        data = np.insert(data, 1, new_i_subj, axis=1)

    return data


def export_group_data_2d():

    conf = ss_timing_analysis.conf.get_conf()

    # trials x (subj, lin_subj, onset, ori, trial, contrast, resp)
    data = load_group_data_2d(exclude=True)

    csv_path = os.path.join(
        conf.group_data_path,
        "ss_timing_subj_resp_data_2d.csv"
    )

    dem = ss_timing_analysis.dem.demographics()

    with open(csv_path, "wb") as csv_file:

        writer = csv.writer(csv_file)

        header = [
            "i_subj_all",
            "i_subj_curr",
            "olife_total",
            "i_onset",
            "i_ori",
            "i_trial",
            "contrast",
            "correct"
        ]

        writer.writerow(header)

        for data_row in data:

            subj_id = conf.all_subj_ids[data_row[0].astype("int")]
            olife_total = dem[subj_id]["olife_total"]

            row = [int(data_row[0]), int(data_row[1]), olife_total]
            row.extend([int(data_item) for data_item in data_row[2:5]])
            row.append(data_row[5])  # contrast, as float
            row.append(int(data_row[6]))

            writer.writerow(row)


def export_scatter_data():

    conf = ss_timing_analysis.conf.get_conf()

    fit_data = ss_timing_analysis.group_fit.load_fit_data(exclude=True)

    alphas = fit_data[0][..., 0, 0]

    subscales = {
        subscale: ss_timing_analysis.dem.get_olife_subscale(
            subscale=subscale, exclude=True
        )
        for subscale in conf.subscales
    }

    csv_path = os.path.join(
        conf.group_data_path,
        "ss_timing_scatter_export.csv"
    )

    with open(csv_path, "w") as csv_file:

        header = conf.surr_onsets + conf.subscales

        csv_file.write(",".join(header) + "\n")

        for i_subj in xrange(alphas.shape[0]):

            row = []

            for (i_onset, surr_onset) in enumerate(conf.surr_onsets):
                row.append(
                    alphas[i_subj, i_onset, 0] - alphas[i_subj, i_onset, 1]
                )

            for subscale in conf.subscales:
                row.append(subscales[subscale][i_subj])

            row_str = map(str, row)

            csv_file.write(",".join(row_str) + "\n")


    return alphas, subscales





def bin_group_data():

    conf = ss_timing_analysis.conf.get_conf()

    data = load_group_data()

    bin_data = np.zeros(
        (
            conf.n_all_subj,
            conf.n_surr_onsets,
            conf.n_surr_oris,
            conf.n_bins,
            2  # (resp prop, n trials)
        )
    )

    for i_s in xrange(conf.n_all_subj):

        for i_onset in xrange(conf.n_surr_onsets):
            for i_ori in xrange(conf.n_surr_oris):

                # return the index of the bin for each trial
                i_bins = np.digitize(
                    data[i_s, i_onset, i_ori, :, 0],
                    conf.bin_edges
                )

                for i_bin in xrange(len(conf.bin_edges)):

                    in_bin = (i_bins == i_bin)

                    n_in_bin = np.sum(in_bin)

                    bin_data[i_s, i_onset, i_ori, i_bin, 1] = n_in_bin

                    # number of '1's in this bin
                    bin_resp = np.sum(
                        data[i_s, i_onset, i_ori, in_bin, 1]
                    )

                    if n_in_bin > 0:
                        bin_data[i_s, i_onset, i_ori, i_bin, 0] = (
                            bin_resp / float(n_in_bin)
                        )

    npy_path = os.path.join(
        conf.group_data_path,
        "ss_timing_all_subj_resp_data_binned.npy"
    )

    np.save(npy_path, bin_data)

    return bin_data
