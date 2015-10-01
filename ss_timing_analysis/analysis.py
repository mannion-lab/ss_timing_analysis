
import numpy as np
import scipy.optimize
import scipy.stats

import ss_timing.conf
import ss_timing.data

import ss_timing_analysis.conf


def get_subj_psi_alpha(conf):

    raw_data = ss_timing.data.load_data(conf)

    data = np.empty(
        (
            conf.n_surr_onsets,
            conf.n_surr_oris,
            conf.n_stairs_per_run * conf.n_runs_per_cond
        )
    )
    data.fill(np.NAN)

    cond_count = np.zeros((conf.n_surr_onsets, conf.n_surr_oris))

    for trial_data in raw_data:

        if trial_data["stair_trial"] == conf.n_trials_per_stair:

            i_onset = trial_data["i_surr_onset"]
            i_ori = trial_data["i_surr_ori"]

            alpha = trial_data["alpha_hat"]

            i_cond = cond_count[i_onset, i_ori]

            data[i_onset, i_ori, i_cond] = alpha

            cond_count[i_onset, i_ori] += 1

    assert np.sum(np.isnan(data))

    return data


def get_all_fit_params():

    conf = ss_timing_analysis.conf.get_conf()

    fit_params = np.empty(
        (
            conf.n_subj,
            conf.n_surr_onsets,
            conf.n_surr_oris,
            2
        )
    )
    fit_params.fill(np.NAN)

    for (i_subj, subj_id) in enumerate(conf.subj_ids):

        conf.subj_id = subj_id

        fit_params[i_subj, ...] = get_fit_params(conf)

    assert np.sum(np.isnan(fit_params)) == 0

    return fit_params


def get_fit_params(conf):

    # onsets x oris x trials x [contrast, correct]
    data = get_subj_resp_data(conf)

    fit_params = np.empty(
        (
            conf.n_surr_onsets,
            conf.n_surr_oris,
            2  # alpha, beta
        )
    )
    fit_params.fill(np.NAN)

    for i_surr_onset in xrange(conf.n_surr_onsets):
        for i_surr_ori in xrange(conf.n_surr_oris):

            f = fit_cond(conf, data[i_surr_onset, i_surr_ori, ...])

            fit_params[i_surr_onset, i_surr_ori, :] = f

    assert np.sum(np.isnan(fit_params)) == 0

    return fit_params


def fit_cond(conf, cond_data):

    p0 = [scipy.stats.gmean(conf.x_levels), 2.0]

    alpha_bounds = [conf.x_levels[0], conf.x_levels[-1]]

    beta_bounds = [0.0, None]

    fit = scipy.optimize.minimize(
        loglike,
        p0,
        (cond_data, conf),
        method="L-BFGS-B",
        bounds=[alpha_bounds, beta_bounds]
    )

    fit_params = fit.x

    return fit_params


def loglike(params, cond_data, conf):

    (alpha, beta) = params

    p = conf.psych_func(
        cond_data[:, 0],
        alpha=alpha,
        beta=beta
    )

    try:
        assert np.sum(np.isnan(p)) == 0
    except:
        print alpha, beta

    resp_was_incorrect = (cond_data[:, 1] == 0)

    p[resp_was_incorrect] = 1 - p[resp_was_incorrect]

    return -np.sum(np.log(p))
