
import numpy as np
import scipy.optimize
import scipy.stats


def fit_ml_boot_subj(data, conf, n_boot=10000):

    fit_params = np.empty(
        (
            conf.n_surr_onsets,
            conf.n_surr_oris,
            2,  # alpha, beta
            n_boot + 1
        )
    )
    fit_params.fill(np.NAN)

    n_trials = data.shape[-2]

    # first pass, no bootstrapping
    i_trials = np.arange(n_trials)

    for i_boot in xrange(n_boot + 1):

        fit = fit_data_ml(
            data=data[..., i_trials, :],
            conf=conf
        )

        fit_params[..., 0, i_boot] = fit[..., 0]
        fit_params[..., 1, i_boot] = fit[..., 1]

        # randomly sample trials, with replacement
        i_trials = np.random.choice(
            a=range(n_trials),
            size=n_trials,
            replace=True
        )

    assert np.sum(np.isnan(fit_params) == 0)

    return fit_params


def fit_data_ml(data, conf):

    fit_params = np.empty(
        (
            conf.n_surr_onsets,
            conf.n_surr_oris,
            2
        )
    )
    fit_params.fill(np.NAN)

    for i_surr_onset in xrange(conf.n_surr_onsets):
        for i_surr_ori in xrange(conf.n_surr_oris):

            fit = fit_cond_ml(data[i_surr_onset, i_surr_ori, ...], conf)

            fit_params[i_surr_onset, i_surr_ori, :] = fit

    return fit_params


def fit_cond_ml(data, conf):

    p0 = [scipy.stats.gmean(conf.x_levels), 2.0]

    alpha_bounds = [conf.x_levels[0], conf.x_levels[-1]]

    beta_bounds = [0.0, None]

    fit = scipy.optimize.minimize(
        loglike,
        p0,
        (data, conf),
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

    resp_was_incorrect = (cond_data[:, 1] == 0)

    p[resp_was_incorrect] = 1 - p[resp_was_incorrect]

    return -np.sum(np.log(p))
