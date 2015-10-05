import os

import numpy as np

import ss_timing_analysis.conf
import ss_timing_analysis.data


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
