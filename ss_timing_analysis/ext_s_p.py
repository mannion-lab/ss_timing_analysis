
import os

import numpy as np

import xlrd
import savReaderWriter

import ss_timing_analysis.conf


def load_data():

    conf = ss_timing_analysis.conf.get_conf()

    data_path = os.path.join(
        conf.base_path,
        "s_p_data",
        "Data_Serrano-PedrazaEA2014.xlsx"
    )

    ws = xlrd.open_workbook(data_path).sheets()[0]

    h = "Contrast detection thresholds. Serrano-Pedraza et al. (2014) Frontier in Psychology (5)"

    # check the header is correct
    assert ws.cell(0, 0).value == h

    n_hc = 24
    hc_data = np.empty((n_hc, 3))
    hc_data.fill(np.NAN)
    i_row_start_hc = 3

    n_sz = 21
    sz_data = np.empty((n_sz, 3))
    sz_data.fill(np.NAN)
    i_row_start_sz = 30

    i_cols = [
        ("ns", 2),
        ("os", 3),
        ("ps", 4)
    ]

    for i_hc in xrange(n_hc):

        i_subj_row = i_row_start_hc + i_hc

        subj_id = ws.cell(i_subj_row, 0).value

        # make sure the subject ID is as expected
        assert subj_id == "C{n:02d}".format(n=i_hc + 1)

        for (i_cond, (_, i_col)) in enumerate(i_cols):

            datum = float(ws.cell(i_subj_row, i_col).value)

            hc_data[i_hc, i_cond] = datum

    for i_sz in xrange(n_sz):

        i_subj_row = i_row_start_sz + i_sz

        subj_id = ws.cell(i_subj_row, 0).value

        # make sure the subject ID is as expected
        assert subj_id == "SZ{n:02d}".format(n=i_sz + 1)

        for (i_cond, (_, i_col)) in enumerate(i_cols):

            datum = float(ws.cell(i_subj_row, i_col).value)

            sz_data[i_sz, i_cond] = datum

    assert np.sum(np.isnan(hc_data)) == 0
    assert np.sum(np.isnan(sz_data)) == 0

    return (hc_data, sz_data)


def spss_export():

    conf = ss_timing_analysis.conf.get_conf()

    (hc_data, sz_data) = load_data()

    sav_path = os.path.join(
        conf.base_path,
        "s_p_data",
        "s_p_data_for_spss.sav"
    )

    var_names = ["GR", "NS", "OS", "PS"]
    var_types = {
        "GR": 2,
        "NS": 0,
        "OS": 0,
        "PS": 0
    }

    with savReaderWriter.SavWriter(sav_path, var_names, var_types) as writer:

        for i_hc in xrange(hc_data.shape[0]):

            row = ["HC"] + list(hc_data[i_hc, :])

            writer.writerow(row)

        for i_sz in xrange(sz_data.shape[0]):

            row = ["SZ"] + list(sz_data[i_sz, :])

            writer.writerow(row)



