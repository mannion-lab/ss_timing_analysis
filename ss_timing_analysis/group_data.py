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


def load_group_data():

    conf = ss_timing_analysis.conf.get_conf()

    npy_path = os.path.join(
        conf.group_data_path,
        "ss_timing_all_subj_resp_data.npy"
    )

    data = np.load(npy_path)

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
