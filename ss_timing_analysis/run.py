import argparse

import ss_timing_analysis.group_data
import ss_timing_analysis.group_fit


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
        "bin_group_data",
        "fit_data"
    ]

    parser.add_argument(
        "analysis",
        help="Analysis to run",
        choices=anas
    )

    args = parser.parse_args()

    if args.analysis == "group_data":
        ss_timing_analysis.group_data.save_group_data()

    elif args.analysis == "bin_group_data":
        ss_timing_analysis.group_data.bin_group_data()

    elif args.analysis == "fit_data":
        ss_timing_analysis.group_fit.fit_data()
