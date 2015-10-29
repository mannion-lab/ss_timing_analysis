import os
import csv
import collections

import numpy as np

import ss_timing_analysis.conf


def export_demographics():

    conf = ss_timing_analysis.conf.get_conf()

    dem = demographics(conf)

    data_path = os.path.join(
        conf.group_data_path,
        "ss_timing_dem.csv"
    )

    with open(data_path, "wb") as data_file:

        writer = csv.writer(data_file)

        header = [
            "subj_id",
            "testing_booth",
            "age",
            "gender",
            "handedness",
            "olife_total"
        ]

        subscales = ["un_ex", "cog_dis", "int_anh", "imp_non"]

        header.extend(
            [
                "olife_{ss:s}".format(ss=subscale)
                for subscale in subscales
            ]
        )

        writer.writerow(header)

        for (subj_id, subj_info) in dem.iteritems():

            row = [
                subj_id,
                subj_info["testing_booth"],
                subj_info["age"],
                subj_info["gender"],
                subj_info["handedness"],
                subj_info["olife_total"]
            ]

            row.extend(
                [
                    subj_info[subscale]["score"]
                    for subscale in subscales
                ]
            )

            writer.writerow(row)


def print_demographics(exclude=True):

    conf = ss_timing_analysis.conf.get_conf()

    dem = demographics(conf)

    if exclude:
        subj_ids = conf.subj_ids
    else:
        subj_ids = conf.all_subj_ids

    n = len(subj_ids)

    ###
    print "Age:"

    ages = [
        int(dem[subj_id]["age"])
        for subj_id in subj_ids
    ]

    age_counts = collections.Counter(ages)

    for (age, count) in age_counts.iteritems():
        print "\t{a:d}: {c:d}/{n:d}".format(
            a=age, c=count, n=n
        )

    ###
    print "Gender:"

    genders = [
        dem[subj_id]["gender"]
        for subj_id in subj_ids
    ]

    gender_counts = collections.Counter(genders)

    for (gender, count) in gender_counts.iteritems():
        print "\t{g:s}: {c:d}/{n:d}".format(
            g=gender, c=count, n=n
        )

    ###
    print "Handedness:"

    hands = [
        dem[subj_id]["handedness"]
        for subj_id in subj_ids
    ]

    hand_counts = collections.Counter(hands)

    for (hand, count) in hand_counts.iteritems():
        print "\t{h:s}: {c:d}/{n:d}".format(
            h=hand, c=count, n=n
        )


def print_olt_descriptives(exclude=True):

    conf = ss_timing_analysis.conf.get_conf()

    dem = demographics(conf)

    if exclude:
        subj_ids = conf.subj_ids
    else:
        subj_ids = conf.all_subj_ids

    sz = [
        dem[subj_id]["olife_total"]
        for subj_id in subj_ids
    ]

    print "Mean: {m:.3f}".format(m=np.mean(sz))
    print "Std: {s:.3f}".format(s=np.std(sz, ddof=1))
    print "Min: {m:.3f}".format(m=np.min(sz))
    print "Max: {m:.3f}".format(m=np.max(sz))


def demographics():

    conf = ss_timing_analysis.conf.get_conf()

    raw_csv_path = os.path.join(
        conf.demographics_path,
        "ss_timing_raw_dem.csv"
    )

    dem = collections.OrderedDict()

    with open(raw_csv_path, "rb") as raw_csv:

        reader = csv.reader(raw_csv)

        header = reader.next()

        for subj_data in reader:

            subj_id = subj_data[header.index("subj_id")]

            dem[subj_id] = {}

            for param in ("testing_booth", "age", "gender", "handedness"):

                dem[subj_id][param] = subj_data[header.index(param)]

            o_life = [
                subj_data[header.index("olife_q{n:d}".format(n=q_num))]
                for q_num in xrange(1, 105)
            ]

            subscales = _score_olife(o_life)

            dem[subj_id]["olife_total"] = 0

            for (subscale_name, subscale_data) in subscales.iteritems():

                dem[subj_id][subscale_name] = subscale_data

                dem[subj_id]["olife_total"] += subscale_data["score"]

    return dem


def _score_olife(data):

    assert len(data) == 104

    subscales = {
        "un_ex": {
            "questions": range(1, 31),
            "neg_questions": []
        },
        "cog_dis": {
            "questions": range(31, 55),
            "neg_questions": []
        },
        "int_anh": {
            "questions": range(55, 82),
            "neg_questions": [
                1, 4, 5, 8, 12, 13, 14, 15, 17, 20, 21, 22, 23, 24, 26
            ]
        },
        "imp_non": {
            "questions": range(82, 105),
            "neg_questions": [6, 10, 15, 17, 19, 23]
        }
    }

    scores = {}

    for (subscale_name, subscale_info) in subscales.iteritems():

        sub = {}

        subscale_data = [
            data[subscale_q_num - 1]
            for subscale_q_num in subscale_info["questions"]
        ]

        sub["orig_data"] = list(subscale_data)

        # check that each response is something that we expect
        for unique_response in set(subscale_data):
            assert unique_response in ("Y", "N", "A", "B")

        # apply negation to the appropriate questions
        subscale_data = _neg_questions(
            subscale_data,
            subscale_info["neg_questions"]
        )

        sub["neg_data"] = list(subscale_data)

        if "A" in subscale_data or "B" in subscale_data:

            # find the modal answer
            if subscale_data.count("Y") > subscale_data.count("N"):
                modal_answer = "Y"
            else:
                modal_answer = "N"

            i_bad_questions = [
                i_bad_q
                for i_bad_q in xrange(len(subscale_data))
                if subscale_data[i_bad_q] in ("A", "B")
            ]

            for i_bad_q in i_bad_questions:
                subscale_data[i_bad_q] = modal_answer

        sub["final_data"] = list(subscale_data)

        sub["score"] = subscale_data.count("Y")

        scores[subscale_name] = sub

    return scores


def _neg_questions(subscale_data, neg_q_nums):

    for neg_q_num in neg_q_nums:

        i_neg_q = neg_q_num - 1

        if subscale_data[i_neg_q] == "Y":
            subscale_data[i_neg_q] = "N"
        elif subscale_data[i_neg_q] == "N":
            subscale_data[i_neg_q] = "Y"

    return subscale_data
