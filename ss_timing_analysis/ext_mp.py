import os

import numpy as np
import scipy.io

import ss_timing_analysis.conf


def load_data():

    conf = ss_timing_analysis.conf.get_conf()

    mat_path = os.path.join(
        conf.base_path,
        "mp_data",
        "ODSS_data_for_Damien_20151206.mat"
    )

    mat_data = scipy.io.loadmat(
        file_name=mat_path,
        squeeze_me=True,
        struct_as_record=False
    )["perceived_contrast"]

    subj_nums = mat_data.subj_idx[:, 0]
    group_nums = mat_data.group_idx[:, 0]

    all_data = mat_data.data

    c_subj_nums = np.unique(subj_nums[group_nums == 1])

    n_c = len(c_subj_nums)

    c_data = np.full((n_c, 2), np.nan)

    for (i_c_subj, c_num) in enumerate(c_subj_nums):

        i_c = (subj_nums == c_num)

        c_data[i_c_subj, :] = np.nanmean(all_data[i_c, :], axis=0)

    p_subj_nums = np.unique(subj_nums[group_nums == 2])

    n_p = len(p_subj_nums)

    p_data = np.full((n_p, 2), np.nan)

    for (i_p_subj, p_num) in enumerate(p_subj_nums):

        i_p = (subj_nums == p_num)

        p_data[i_p_subj, :] = np.nanmean(all_data[i_p, :], axis=0)

    return c_data, p_data
