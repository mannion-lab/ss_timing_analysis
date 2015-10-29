import argparse
import sys

import logging
log = logging.getLogger()
log.setLevel(logging.INFO)
screen_handler = logging.StreamHandler(sys.stdout)
log.addHandler(screen_handler)


import ss_timing_analysis.group_data
import ss_timing_analysis.group_fit
import ss_timing_analysis.figures
import ss_timing_analysis.dem


def main():
    "Parse the command-line input and offload"

    description = "Analysis commands for ss_timing"

    fmt = argparse.ArgumentDefaultsHelpFormatter

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=fmt
    )

    anas = [
        "group_data",
        "group_data_2d",
        "export_group_data_2d",
        "export_dem",
        "print_dem",
        "bin_group_data",
        "fit_data",
        "subjects_figure",
        "eg_subject_figure",
        "thresholds",
        "sim_scatter"
    ]

    parser.add_argument(
        "analysis",
        help="Analysis to run",
        choices=anas
    )

    args = parser.parse_args()

    if args.analysis == "group_data":
        ss_timing_analysis.group_data.save_group_data()

    elif args.analysis == "group_data_2d":
        ss_timing_analysis.group_data.save_group_data_2d()

    elif args.analysis == "export_group_data_2d":
        ss_timing_analysis.group_data.export_group_data_2d()

    elif args.analysis == "export_dem":
        ss_timing_analysis.dem.export_demographics()

    elif args.analysis == "print_dem":
        ss_timing_analysis.dem.print_demographics()
        ss_timing_analysis.dem.print_olt_descriptives()

    elif args.analysis == "bin_group_data":
        ss_timing_analysis.group_data.bin_group_data()

    elif args.analysis == "fit_data":
        ss_timing_analysis.group_fit.fit_data()

    elif args.analysis == "subjects_figure":
        ss_timing_analysis.figures.subjects(save_pdf=True)

    elif args.analysis == "eg_subject_figure":
        ss_timing_analysis.figures.eg_subject(save_pdf=True)

    elif args.analysis == "thresholds":
        ss_timing_analysis.figures.thresholds(save_pdf=True)

    elif args.analysis == "sim_scatter":
        ss_timing_analysis.figures.sim_scatter(save_pdf=True)


