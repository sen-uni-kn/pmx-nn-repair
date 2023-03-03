import argparse
import pandas as pd
import ruamel.yaml as yaml

import torch
import scipy.io


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble Data on an IDR Network for Plotting"
    )
    parser.add_argument(
        "network_info_file",
        help="The .yaml network info file of the trained or repaired network."
    )
    parser.add_argument("--lognormal", help="The path of a lognormal dataset.")
    parser.add_argument("--uniform", help="The path of a uniform dataset.")
    parser.add_argument("--grid", help="The path of the grid.")
    parser.add_argument("--out", help="The .mat output file")

    args = parser.parse_args()
    with open(args.network_info_file, "rt") as file:
        network_info = yaml.safe_load(file)
    if "training_info" in network_info:
        network = torch.load(network_info["network"]["repaired_network_path"])
        training_info_file = network_info["training_info"]
        with open(training_info_file) as file:
            training_info = yaml.safe_load(file)
    else:
        training_info = network_info
        network = torch.load(network_info["network"]["trained_network_path"])
    dataset_info = training_info["dataset"]

    lognormal_dataset = pd.read_csv(args.lognormal)
    uniform_dataset = pd.read_csv(args.uniform)
    train_end = dataset_info["training_set_size"]
    val_end = train_end + dataset_info["validation_set_size"]
    lognormal_test_set = torch.as_tensor(
        lognormal_dataset.iloc[val_end:].to_numpy(), dtype=torch.float
    )
    uniform_test_set = torch.as_tensor(
        lognormal_dataset.iloc[val_end:].to_numpy(), dtype=torch.float
    )

    grid_dataset = pd.read_csv(args.grid)
    grid = torch.as_tensor(
        grid_dataset.to_numpy(), dtype=torch.float
    )

    out_data = {}
    for key, data in (
        ("lognormal", lognormal_test_set), ("uniform", uniform_test_set), ("grid", grid)
    ):
        inputs, outputs = data[:, :-1], data[:, -1]
        preds = network(inputs).squeeze()
        out_data[key] = {
            "t": inputs[:, -1].tolist(),
            "c": outputs.tolist(),
            "c_NN": preds.tolist()
        }

    scipy.io.savemat(args.out, out_data)
