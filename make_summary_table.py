import argparse
import sys
from glob import glob
from pathlib import Path

import pandas as pd
from ruamel import yaml
from natsort import natsorted, natsort_key


def at_keys(nested, keys):
    """
    Access an element at (key1, key2, ...) in a
    nested dictionary :code:`nested`.
    """
    val = nested
    for key in keys:
        val = val[key]
    return val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a summary table for an output directory."
    )
    parser.add_argument("files", nargs="+", help="The path to the output directory")
    parser.add_argument("--out", default=None, help="Where to store the summary table.")

    args = parser.parse_args()
    results = []
    for file_path in args.files:
        # skip non-yaml files
        if not file_path.endswith(".yaml"):
            continue
        with open(file_path, 'rt') as file:
            result_dict = yaml.safe_load(file)
            results.append(result_dict)

    # for repair results, add dataset and network information from the
    # training run
    for result_dict in results:
        if "training_info" in result_dict:
            with open(result_dict["training_info"], "rt") as file:
                train_result_dict = yaml.safe_load(file)
                result_dict["dataset"] = train_result_dict["dataset"]
                for key, value in train_result_dict["network"].items():
                    if key not in result_dict["network"]:
                        result_dict["network"][key] = value

    # find the keys that change in the result dictionaries
    changing_keys = []
    for top_key in ("dataset", "network"):
        for key in results[0][top_key]:
            # exclude keys that end in "path"
            if key.endswith("path"):
                continue
            values = [results_dict[top_key][key] for results_dict in results]
            if len(set(values)) > 1:
                changing_keys.append((top_key, key))
    changing_keys = natsorted(changing_keys, key=lambda keys: keys[-1])

    # the changing keys in combination are the rows of the table
    table = []
    for result_dict in results:
        row = {keys[-1]: at_keys(result_dict, keys) for keys in changing_keys}
        row.update(result_dict["results"])
        table.append(row)

    table = pd.DataFrame(table)
    table = table.sort_values(
        by=[keys[-1] for keys in changing_keys],
        key=natsort_key,
    )
    if args.out is None:
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):
            print(table)
    else:
        with open(args.out, "w+t") as file:
            table.to_csv(file, index=False)
