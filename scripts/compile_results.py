#!/usr/bin/env python3
"""
Script to compile results from individual csv files
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

models = [
    "modal",
    "all_parallel",
    "noise_parallel_transient_params",
    "noise_params",
    "noise_transient_params",
    "transient_params",
]


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "indir", help="Input dir -- root log dir for metric csvs", type=str
    )
    parser.add_argument("type", help="Table type: ['all', 'instrument']", type=str)
    parser.add_argument(
        "-o",
        "--outfile",
        help="Output file",
        default=sys.stdout,
        type=argparse.FileType("w"),
    )

    args = parser.parse_args(arguments)

    indir = Path(args.indir)

    # Get all csv files in indir
    csv_files = [f for f in indir.rglob("metrics.csv") if f.is_file()]

    rows = []
    for f in csv_files:
        sub_df = pd.read_csv(f)
        version = f.parent.name
        assert version == "version_0"

        model_name = f.parent.parent.name
        for model in models:
            if model_name.startswith(model):
                _, split = model_name.split(model)
                split = split[1:]

                sub_df["instrument"] = "all"
                if split == "a":
                    sub_df["source"] = "acoustic"
                elif split == "e":
                    sub_df["source"] = "electronic"
                elif split == "all":
                    sub_df["source"] = "all"
                else:
                    sub_df["instrument"] = split
                    sub_df["source"] = "all"

                sub_df["model"] = model

        rows.append(sub_df)

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(by=["model", "source", "instrument"])

    if args.type == "all":
        df = df[df["instrument"] == "all"]
    elif args.type == "instrument":
        df = df[df["source"] == "all"]
        df = df[df["instrument"] != "all"]
    else:
        raise ValueError(f"Unknown type {args.type}")

    print(df)

    rows = []

    # Loop through models
    for model in models:
        model_df = df[df["model"] == model]
        model_row = {"model": [model]}

        if args.type == "all":
            for source in ["all", "acoustic", "electronic"]:
                source_df = model_df[model_df["source"] == source]
                model_row[source + "_mss"] = [source_df["test/loss"].mean()]
                model_row[source + "_lsd"] = [source_df["test/lsd"].mean()]
                model_row[source + "_flux_onset"] = [
                    source_df["test/flux_onset"].mean()
                ]
        else:
            for inst in ["kick", "snare", "tom", "cymbals"]:
                inst_df = model_df[model_df["instrument"] == inst]
                model_row[inst + "_mss"] = [inst_df["test/loss"].mean()]
                model_row[inst + "_lsd"] = [inst_df["test/lsd"].mean()]
                model_row[inst + "_flux_onset"] = [inst_df["test/flux_onset"].mean()]

        rows.append(pd.DataFrame.from_dict(model_row))

    df = pd.concat(rows, ignore_index=True)

    def float_format(x):
        if x < 10 and x > 0.1:
            return "%.2f" % x
        elif x < 0.1:
            return "%.3f" % x
        else:
            return "%.1f" % x

    df.to_latex(args.outfile, index=False, float_format=float_format)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
