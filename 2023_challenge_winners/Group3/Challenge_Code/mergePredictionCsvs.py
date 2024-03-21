import os
import argparse
import numpy as np
import pandas as pd
from ast import literal_eval


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-csv", type=str, required=True)
    parser.add_argument("--task-csv", type=str, required=True)
    parser.add_argument("--results-csv", type=str, default="Group_3_results.csv")
    
    args = parser.parse_args()
    return args


def merge(bbox_csv=None, task_csv=None, results_csv=None):
    bbox_df = pd.read_csv(bbox_csv)
    task_df = pd.read_csv(task_csv)

    # Convert tool bounding box column to list of dicts
    bbox_df["Tool bounding box"] = bbox_df["Tool bounding box"].apply(literal_eval)

    # Remove label columns from task csv
    cols_to_drop = ["Cross-section", "Long-axis", "anesthetic", "dilator", "insert_catheter",
                    "insert_guidewire", "insert_needle", "nothing", "remove_guidewire", "scalpel"]
    task_df.drop(columns=cols_to_drop, inplace=True)

    # Left join on FileName column
    df = pd.merge(task_df, bbox_df, how="left", on="FileName")
    df["Tool bounding box"] = df["Tool bounding box"].apply(lambda x: x if isinstance(x, list) else [])

    assert len(task_df.index) == len(df.index)

    # Save submission
    df.to_csv(results_csv, index=False)
    print(f"Saved results csv file to {results_csv}.")


def main(args):
    merge(**vars(args))


if __name__ == "__main__":
    args = get_parser()
    main(args)
