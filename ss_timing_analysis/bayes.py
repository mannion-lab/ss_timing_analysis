
import numpy as np

import pymc as pm

import ss_timing_analysis.conf
import ss_timing_analysis.group_fit


def model():

    conf = ss_timing_analysis.conf.get_conf()

    data = ss_timing_analysis.group_data.load_group_data_2d(
        exclude=False
    )

    i_subj = data[:, 0].astype("int")
    i_onset = data[:, 1].astype("int")
    i_ori = data[:, 2].astype("int")

    contrasts = data[:, -2]
    responses = data[:, -1]

    # o-life total
    sz = np.array(
        [
            conf.demographics[subj_id]["olt"]
            for subj_id in conf.all_subj_ids
        ]
    )

    # mean-centre the schizotypy score
    # a good idea?
    sz = sz - np.mean(sz)
    sz = sz[:, np.newaxis, np.newaxis]

    info = [
        {
            "name": "bS",
            "size": (conf.n_all_subj, 1, 1)
        },
        {
            "name": "bD",
            "size": (1, conf.n_surr_onsets, 1)
        },
        {
            "name": "bT",
            "size": (1, 1, conf.n_surr_oris)
        },
        {
            "name": "bDxT",
            "size": (1, conf.n_surr_onsets, conf.n_surr_oris)
        },
        {
            "name": "bDxZ",
            "size": (1, conf.n_surr_onsets, 1)
        },
        {
            "name": "bTxZ",
            "size": (1, 1, conf.n_surr_oris)
        },
        {
            "name": "bDxTxZ",
            "size": (1, conf.n_surr_onsets, conf.n_surr_oris)
        }
    ]

    b0_mu = {
        "thresh": np.log(0.25),
        "slope": 1.0
    }

    b0_tau = {
        "thresh": np.log(2) ** -2.0,
        "slope": 10.0 ** -2.0
    }

    b0_val = {
        "thresh": np.log(0.1),
        "slope": 1.0
    }

    params = {}

    for p_type in ("thresh", "slope"):

        params[p_type + "_b0"] = pm.Normal(
            p_type + "_b0",
            mu=b0_mu[p_type],
            tau=b0_tau[p_type],
            value=b0_val[p_type]
        )

        params[p_type + "_bZ"] = pm.Normal(
            p_type + "_bZ",
            mu=0.0,
            tau=0.001,
            value=0.0
        )

        for param_info in info:

            sig_name = p_type + "_" + param_info["name"] + "_sigma"

            params[sig_name] = pm.Uniform(
                sig_name,
                lower=0.0,
                upper=10.0,
                value=0.1
            )

            norm_name = p_type + "_" + param_info["name"]

            params[norm_name] = pm.Normal(
                norm_name,
                mu=0.0,
                tau=params[sig_name] ** -2.0,
                size=param_info["size"]
            )

    @pm.deterministic
    def thresh(params=params, sz=sz):

        return (
            params["thresh_b0"] +
            params["thresh_bZ"] * sz +
            params["thresh_bS"] +
            params["thresh_bD"] +
            params["thresh_bT"] +
            params["thresh_bDxT"] +
            params["thresh_bDxZ"] * sz +
            params["thresh_bTxZ"] * sz +
            params["thresh_bDxTxZ"] * sz
        )

    @pm.deterministic
    def slope(params=params, sz=sz):

        return (
            params["slope_b0"] +
            params["slope_bZ"] * sz +
            params["slope_bS"] +
            params["slope_bD"] +
            params["slope_bT"] +
            params["slope_bDxT"] +
            params["slope_bDxZ"] * sz +
            params["slope_bTxZ"] * sz +
            params["slope_bDxTxZ"] * sz
        )

    @pm.deterministic
    def psych_func(
        thresh=thresh,
        slope=slope,
        contrasts=contrasts,
        i_onset=i_onset,
        i_ori=i_ori,
        conf=conf
    ):

        # threshold is in log units, slope in linear
        thresh = np.exp(thresh)

        return conf.psych_func(
            contrasts,
            thresh[i_subj, i_onset, i_ori],
            slope[i_subj, i_onset, i_ori]
        )

    obs = pm.Bernoulli(
        "obs",
        p=psych_func,
        observed=True,
        value=responses
    )

    param_list = params.values()
    param_list.extend(
        [
            psych_func,
            obs,
            thresh,
            slope
        ]
    )

    model = pm.Model(param_list)

    return model
