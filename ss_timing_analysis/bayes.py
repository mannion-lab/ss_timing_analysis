
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

    thresh_a0 = pm.Normal(
        "thresh_a0",
        mu=np.log(0.25),
        tau=np.log(2) ** -2,
        trace=False,
        plot=False,
        value=np.log(0.25)
    )

    # deflection due to schizotypy
    thresh_bZ = pm.Normal(
        "thresh_bZ",
        mu=0.0,
        tau=0.001,
        trace=False,
        plot=False,
        value=0.0
    )

    thresh_params = [
        {
            "name": "aS",
            "size": (conf.n_all_subj, 1, 1)
        },
        {
            "name": "aD",
            "size": (1, conf.n_surr_onsets, 1)
        },
        {
            "name": "aT",
            "size": (1, 1, conf.n_surr_oris)
        },
        {
            "name": "aSxD",
            "size": (conf.n_all_subj, conf.n_surr_onsets, 1)
        },
        {
            "name": "aSxT",
            "size": (conf.n_all_subj, 1, conf.n_surr_oris)
        },
        {
            "name": "aDxT",
            "size": (1, conf.n_surr_onsets, conf.n_surr_oris)
        },
        {
            "name": "aSxDxT",
            "size": (conf.n_all_subj, conf.n_surr_onsets, conf.n_surr_oris)
        },
        {
            "name": "aDxZ",
            "size": (1, conf.n_surr_onsets, 1)
        },
        {
            "name": "aTxZ",
            "size": (1, 1, conf.n_surr_oris)
        },
        {
            "name": "aDxTxZ",
            "size": (1, conf.n_surr_onsets, conf.n_surr_oris)
        }
    ]

    thresh_sigmas = {}
    thresh_normals = {}

    for param_dict in thresh_params:

        sigma_name = "thresh_" + param_dict["name"] + "_sigma"

        thresh_sigmas[sigma_name] = pm.Uniform(
            sigma_name,
            lower=0.0,
            upper=10.0,
            value=0.1
        )

        norm_name = "thresh_" + param_dict["name"]

        thresh_normals[norm_name] = pm.Normal(
            norm_name,
            mu=0.0,
            tau=thresh_sigmas[sigma_name] ** -2.0,
            trace=False,
            plot=False,
            size=param_dict["size"]
        )

    @pm.deterministic(plot=False, trace=True)
    def thresh_cells(
        thresh_a0=thresh_a0,
        thresh_bZ=thresh_bZ,
        thresh_normals=thresh_normals,
        sz=sz
    ):

        return (
            thresh_a0 +
            thresh_bZ * sz +
            thresh_normals["thresh_aS"] +
            thresh_normals["thresh_aD"] +
            thresh_normals["thresh_aT"] +
            thresh_normals["thresh_aSxD"] +
            thresh_normals["thresh_aSxT"] +
            thresh_normals["thresh_aDxT"] +
            thresh_normals["thresh_aSxDxT"] +
            thresh_normals["thresh_aDxZ"] * sz +
            thresh_normals["thresh_aTxZ"] * sz +
            thresh_normals["thresh_aDxTxZ"] * sz
        )


    thresh = np.random.rand(78, 2, 2)
    slope = np.random.rand(78, 2, 2)

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

    params = [
        psych_func,
        obs,
        thresh_cells
    ]
    params.extend(thresh_sigmas.values())
    params.extend(thresh_normals.values())

    model = pm.Model(params)

    return model
