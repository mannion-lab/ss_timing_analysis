
import numpy as np

import ss_timing.conf
import ss_timing.data


def get_subj_resp_data(subj_id):

    conf = ss_timing.conf.get_conf(subj_id)

    raw_data = ss_timing.data.load_data(conf)

    resp_data = np.empty(
        (
            conf.n_surr_onsets,  # [pre, sim]
            conf.n_surr_oris,  # [orth, para]
            (
                conf.n_trials_per_stair *
                conf.n_stairs_per_run *
                conf.n_runs_per_cond
            ),  # trials
            2  # [contrast, correct or not]
        )
    )
    resp_data.fill(np.NAN)

    cond_count = np.zeros((conf.n_surr_onsets, conf.n_surr_oris))

    for trial_data in raw_data:

        i_onset = trial_data["i_surr_onset"]
        i_ori = trial_data["i_surr_ori"]

        correct = trial_data["correct"]
        contrast = trial_data["target_contrast"]

        i_trial = cond_count[i_onset, i_ori]

        resp_data[i_onset, i_ori, i_trial, 0] = contrast
        resp_data[i_onset, i_ori, i_trial, 1] = correct

        cond_count[i_onset, i_ori] += 1

    assert np.sum(np.isnan(resp_data)) == 0

    assert np.all(cond_count == resp_data.shape[2])

    return resp_data
