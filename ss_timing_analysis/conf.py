
import ss_timing.conf


def get_conf(subj_id):

    conf = ss_timing.conf.get_conf(subj_id)

    conf.surr_ori_labels = ["orth", "para"]

    return conf
