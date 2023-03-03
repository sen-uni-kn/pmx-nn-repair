import argparse
import pandas as pd
import ruamel.yaml as yaml

import torch
import scipy.io

from datasets.dataset_1_cmp_po import Dataset1CMPPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble Data on a 1 CMP PO Network for Plotting"
    )
    parser.add_argument(
        "network_info_file",
        help="The .yaml network info file of the trained or repaired network."
    )
    parser.add_argument("--lognormal", help="The path of a lognormal dataset.")
    parser.add_argument("--uniform", help="The path of a uniform dataset.")
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

    grid_points = 5
    dose_min, Cl_min, V_min, ka_min, t_min = network.mins
    dose_max, Cl_max, V_max, ka_max, t_max = network.maxes
    grid_dose = torch.linspace(dose_min, dose_max, grid_points)
    grid_Cl = torch.linspace(Cl_min, Cl_max, grid_points)
    grid_V = torch.linspace(V_min, V_max, grid_points)
    grid_ka = torch.linspace(ka_min, ka_max, grid_points)
    grid_t = torch.linspace(t_min, t_max, dataset_info["samples_per_patient"])
    grid = torch.cartesian_prod(grid_dose, grid_Cl, grid_V, grid_ka, grid_t)
    grid_outputs = Dataset1CMPPO.compute_output(*grid.T).unsqueeze(-1)
    grid = torch.hstack([grid, grid_outputs])

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
