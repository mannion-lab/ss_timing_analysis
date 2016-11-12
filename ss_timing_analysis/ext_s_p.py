
import os

import numpy as np
import scipy.stats

import xlrd
import savReaderWriter

import ss_timing_analysis.conf
import ss_timing_analysis.ext


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

    n_vals = 4

    n_hc = 24
    hc_data = np.empty((n_hc, n_vals))
    hc_data.fill(np.NAN)
    i_row_start_hc = 3
    hc_ages = []

    n_sz = 21
    sz_data = np.empty((n_sz, n_vals))
    sz_data.fill(np.NAN)
    i_row_start_sz = 30
    sz_ages = []

    i_cols = [
        ("ns", 2),
        ("os", 3),
        ("ps", 4)
    ]

    i_age_col = 1

    for i_hc in xrange(n_hc):

        i_subj_row = i_row_start_hc + i_hc

        subj_id = ws.cell(i_subj_row, 0).value

        # make sure the subject ID is as expected
        assert subj_id == "C{n:02d}".format(n=i_hc + 1)

        for (i_cond, (_, i_col)) in enumerate(i_cols):

            datum = float(ws.cell(i_subj_row, i_col).value)

            hc_data[i_hc, i_cond] = datum

        hc_data[i_hc, -1] = float(ws.cell(i_subj_row, i_age_col).value)

    for i_sz in xrange(n_sz):

        i_subj_row = i_row_start_sz + i_sz

        subj_id = ws.cell(i_subj_row, 0).value

        # make sure the subject ID is as expected
        assert subj_id == "SZ{n:02d}".format(n=i_sz + 1)

        for (i_cond, (_, i_col)) in enumerate(i_cols):

            datum = float(ws.cell(i_subj_row, i_col).value)

            sz_data[i_sz, i_cond] = datum

        sz_data[i_sz, -1] = float(ws.cell(i_subj_row, i_age_col).value)

    assert np.sum(np.isnan(hc_data)) == 0
    assert np.sum(np.isnan(sz_data)) == 0

    return (hc_data, sz_data)


def check():

    data = load_data()
    ratios = ss_timing_analysis.ext.get_ratios(data, log_xform=True)

    for (src_str, src, src_r) in zip(("Control", "Patient"), data, ratios):

        print src_str + ":"

        for s in (src, src_r):

            print "\t{m:s}".format(m=", ".join(map(str, np.mean(s, axis=0))))

    # t-tests
    for i_surr in xrange(2):

        (t, p) = scipy.stats.ttest_ind(
            ratios[0][:, i_surr],
            ratios[1][:, i_surr]
        )

        print "t = {t:.2f}, p = {p:.3f}".format(t=t, p=p)

