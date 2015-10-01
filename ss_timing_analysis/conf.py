import os

import xlrd

import ss_timing.conf


def get_conf():

    conf = ss_timing.conf.get_conf("")

    conf.demographics_path = "/home/damien/venv_study/ss_timing/demographics"
    conf.demographics_date = "20151002"

    conf.surr_ori_labels = ["orth", "para"]

    conf.exclude_ids = [
        "p1040"  # started but wrong experiment
    ]

    min_id = 1001
    max_id = 1079

    conf.all_subj_ids = [
        "p{n:d}".format(n=n)
        for n in range(min_id, max_id + 1)
    ]

    conf.subj_ids = [
        subj_id
        for subj_id in conf.all_subj_ids
        if subj_id not in conf.exclude_ids
    ]

    conf.n_all_subj = len(conf.all_subj_ids)
    conf.n_subj = len(conf.subj_ids)

    conf.demographics = demographics(conf)

    return conf


def demographics(conf):

    dem = {}

    xls_path = os.path.join(
        conf.demographics_path,
        "sstimingspreadsheet_" + conf.demographics_date + ".xlsx"
    )

    wb = xlrd.open_workbook(xls_path).sheets()[0]

    keys = [wb.cell(0, i_col).value for i_col in xrange(wb.ncols)]

    for i_row in range(1, wb.nrows):

        subj_id = (
            "p{n:d}".format(
                n=int(wb.cell(i_row, keys.index("Participant ID")).value)
            )
        )

        dem[subj_id] = {
            key: wb.cell(i_row, keys.index(key)).value
            for key in keys
            if key != "Participant ID"
        }

        dem[subj_id]["olt"] = (
            dem[subj_id]["CogDis"] +
            dem[subj_id]["ImpNon"] +
            dem[subj_id]["IntAnh"] +
            dem[subj_id]["UnEx"]
        )

    return dem
