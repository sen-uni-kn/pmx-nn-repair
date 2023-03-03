import argparse
import os
import sys
from datetime import datetime
from functools import partial
import logging
from logging import info
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils import tensorboard
from tqdm import tqdm
import ruamel.yaml as yaml

from datasets.dataset_1_cmp_po import Dataset1CMPPO

from nn_repair.training import (
    TrainingLoop, LogLoss,
    TrainingLossChange, IterationMaximum,
    ValidationSet, TensorboardLossPlot
)

from experiments.experiment_base import seed_rngs
from utils import get_network_factory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 1 CMP PO model.')

    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument("--N", type=int, default=5000, help="Number of patients.")
    dataset_group.add_argument(
        "--samples_per_patient",
        type=int,
        default=25,
        help="Sampled time points per patient."
    )
    dataset_group.add_argument(
        "--Cl", type=str,
        help="Characterize the distribution of the Cl parameter. For example: \n"
             " - log-normal[0.2,0.3] -> log normal distribution with standard deviation "
             "0.3 and mean ln(0.2).\n"
             " - uniform[0.05,0.606] -> uniform distribution with offset 0.05 and "
             "spread 0.606.",
        default="log-normal[0.2,0.3]",
    )
    dataset_group.add_argument(
        "--V", type=str,
        help="Characterize the distribution of the V parameter. See --Cl for examples.",
        default="log-normal[2.0,0.3]",
    )
    dataset_group.add_argument(
        "--ka", type=str,
        help="Characterize the distribution of the V parameter. See --Cl for examples.",
        default="log-normal[0.5,0.3]",
    )
    dataset_group.add_argument(
        "--WT", type=str,
        help="Characterize the distribution of the WT parameter. See --Cl for examples.",
        default="uniform[1.0,3.0]",
    )
    dataset_group.add_argument(
        "--dose", type=float, default=50,
        help="The dose at WT=1."
    )
    dataset_group.add_argument(
        "--t_max", type=float, default=24.0,
        help="The duration of the sampled time frame."
    )
    dataset_group.add_argument(
        "--WT_min", type=float, default=1.0,
        help="The minimum value of WT in the dataset."
    )
    dataset_group.add_argument(
        "--WT_max", type=float, default=4.0,
        help="The minimum value of WT in the dataset."
    )
    dataset_group.add_argument(
        "--Cl_min", type=float, default=0.05,
        help="A lowest accepted value for Cl."
    )
    dataset_group.add_argument(
        "--Cl_max", type=float, default=0.05 + 0.62,
        help="A highest accepted value for Cl."
    )
    dataset_group.add_argument(
        "--V_min", type=float, default=0.63,
        help="A lowest accepted value for V."
    )
    dataset_group.add_argument(
        "--V_max", type=float, default=7.3,
        help="A highest accepted value for V."
    )
    dataset_group.add_argument(
        "--ka_min", type=float, default=0.13,
        help="A lowest accepted value for ka."
    )
    dataset_group.add_argument(
        "--ka_max", type=float, default=0.13 + 1.38,
        help="A highest accepted value for ka."
    )

    training_group = parser.add_argument_group("Neural Network")
    training_group.add_argument(
        '--architecture', type=str,
        help='The network architecture as a list of layer sizes. '
             'For Example: \n'
             ' - 20 -> shallow neural network with 20 neurons\n '
             ' - 10,5 -> neural network with two hidden layers, '
             'the first with size 10, the second with size 5.',
        default='50'
    )
    training_group.add_argument(
        "--batch_size", type=int, default=64,
        help="The mini batch size used for training the network."
    )
    training_group.add_argument(
        "--lr1", type=float, default=0.01,
        help="The learning rate for the first stage of training."
    )
    training_group.add_argument(
        "--lr2", type=float, default=0.001,
        help="The learning rate for second stage of training."
    )
    training_group.add_argument(
        "--optim", choices=["Adam", "RMSprop", "SGD"],
        default="RMSprop",
        help="The training algorithm to use for training."
    )
    training_group.add_argument(
        "--restarts", type=int, default=5,
        help="How many random restarts to perform for training the "
             " network."
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_name", default=None,
        help="A file prefix to prepend before the files that store"
             "the best trained network and further data. "
             "When the output name is None, a timestamp is used."
    )
    output_group.add_argument(
        "--show_plots",
        action="store_true",
        help="Show some plots of the generated dataset and the trained "
             "neural network using tensorboard."
    )

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = (
        f"train_1_cmp_po_"
        f"N_{args.N}_"
        f"samples_per_patient_{args.samples_per_patient}_"
        f"Cl_{args.Cl}_V_{args.V}_ka_{args.ka}_WT_{args.WT}_dose_{args.dose}_"
        f"t_max_{args.t_max}_"
        f"architecture_{args.architecture}_"
        f"batch_size_{args.batch_size}_"
        f"lr1_{args.lr1}_lr2_{args.lr2}_optim_{args.optim}_restarts_{args.restarts}_"
        f"{timestamp}"
    )
    output_name = timestamp if args.output_name is None else args.output_name
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )
    seed_rngs(80324072251433)

    info("Loading Dataset...")
    dataset = Dataset1CMPPO(
        root=Path("output_1_CMP_PO"),
        size=args.N * args.samples_per_patient,
        samples_per_patient=args.samples_per_patient,
        Cl_distribution=args.Cl,
        V_distribution=args.V,
        ka_distribution=args.ka,
        WT_distribution=args.WT,
        dose_at_unit_WT=args.dose,
        t_max=args.t_max,
    )

    dose_min, dose_max = args.dose * args.WT_min, args.dose * args.WT_max
    Cl_min, Cl_max = args.Cl_min, args.Cl_max
    V_min, V_max = args.V_min, args.V_max
    ka_min, ka_max = args.ka_min, args.ka_max
    t_min, t_max = 0.0, args.t_max
    dose, ind_Cl, ind_V, ind_ka, t = dataset.data_in.T
    dose_min_data, dose_max_data = torch.amin(dose), torch.amax(dose)
    Cl_min_data, Cl_max_data = torch.amin(ind_Cl), torch.amax(ind_Cl)
    V_min_data, V_max_data = torch.amin(ind_V), torch.amax(ind_V)
    ka_min_data, ka_max_data = torch.amin(ind_ka), torch.amax(ind_ka)
    t_min_data, t_max_data = torch.amin(t), torch.amax(t)
    info(
        f"Dataset ranges: [min specified | data min - data max | max specified]\n"
        f"dose: [{dose_min:8.4f} | {dose_min_data:8.4f} - "
        f"{dose_max_data:8.4f} | {dose_max:8.4f}]\n"
        f"Cl:   [{Cl_min:8.4f} | {Cl_min_data:8.4f} - "
        f"{Cl_max_data:8.4f} | {Cl_max:8.4f}]\n"
        f"V:    [{V_min:8.4f} | {V_min_data:8.4f} - "
        f"{V_max_data:8.4f} | {V_max:8.4f}]\n"
        f"ka:   [{ka_min:8.4f} | {ka_min_data:8.4f} - "
        f"{ka_max_data:8.4f} | {ka_max:8.4f}]\n"
        f"t:    [{t_min:8.4f} | {t_min_data:8.4f} - "
        f"{t_max_data:8.4f} | {t_max:8.4f}]"
    )
    # check the generated data respects specified bounds
    if dose_min > dose_min_data or dose_max < dose_max_data:
        raise ValueError("Doses (therefore, WT) in dataset out of range!")
    if Cl_min > Cl_min_data or Cl_max < Cl_max_data:
        raise ValueError("Values of Cl in dataset out of range!")
    if V_min > V_min_data or V_max < V_max_data:
        raise ValueError("Values of V in dataset out of range!")
    if ka_min > ka_min_data or ka_max < ka_max_data:
        raise ValueError("Values of ka in dataset out of range!")

    if args.show_plots:
        tensorboard_dir = ".tensorboard" + os.sep + experiment_name
        info(f"Using tensorboard for plotting (results in {tensorboard_dir})")
        tensorboard_writer = tensorboard.writer.SummaryWriter(log_dir=tensorboard_dir)
        tensorboard_writer.add_histogram(
            "Cl", ind_Cl, bins=100,
        )
        tensorboard_writer.add_histogram(
            "V", ind_V, bins=100,
        )
        tensorboard_writer.add_histogram(
            "ka", ind_ka, bins=100,
        )
        tensorboard_writer.add_histogram(
            "dose", dose, bins=100,
        )
        tensorboard_writer.add_histogram(
            "t", t, bins=100,
        )

    # choose 70% of the patients for training, 15% for validation,
    # and 15% for testing
    train_end = int(0.7 * args.N) * args.samples_per_patient
    val_end = train_end + int(0.15 * args.N) * args.samples_per_patient
    train_set = Subset(dataset, indices=list(range(train_end)))
    val_set = Subset(dataset, indices=list(range(train_end, val_end)))
    test_set = Subset(dataset, indices=list(range(val_end, len(dataset))))

    info(
        f"Dataset sizes: train: {len(train_set)} (70%), "
        f"val: {len(val_set)} (15%), test: {len(test_set)} (15%)"
    )

    batch_size = args.batch_size
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    full_train_loader = DataLoader(train_set, batch_size=len(train_set))
    val_loader = DataLoader(val_set, batch_size=len(val_set))
    val_loader_batch = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=len(test_set))

    train_inputs, train_outputs = next(iter(full_train_loader))
    new_network = get_network_factory(
        args.architecture,
        train_inputs,
        train_outputs,
        input_mins=torch.tensor([dose_min, Cl_min, V_min, ka_min, t_min]),
        input_maxs=torch.tensor([dose_max, Cl_max, V_max, ka_max, t_max]),
    )
    network = new_network()

    info(f"Training Network (architecture: {args.architecture})...")
    loss_function = torch.nn.MSELoss()

    def loss(data_loader):
        inputs, targets = next(iter(data_loader))
        pred = network(inputs)
        return loss_function(pred, targets)

    def r_squared(data_loader):
        inputs, targets = next(iter(data_loader))
        pred = network(inputs)
        return 1 - loss_function(pred, targets) / targets.var()

    def violations(data_loader):
        inputs, _ = next(iter(data_loader))
        preds = network(inputs)
        return 100 * (preds < 0.0).any(dim=1).float().mean()

    train_loss = partial(loss, train_loader)
    full_train_loss = partial(loss, full_train_loader)
    val_loss = partial(loss, val_loader)
    test_loss = partial(loss, test_loader)
    val_r_squared = partial(r_squared, val_loader)
    test_r_squared = partial(r_squared, test_loader)
    full_train_r_squared = partial(r_squared, full_train_loader)
    val_viols = partial(violations, val_loader)
    test_viols = partial(violations, test_loader)
    full_train_viols = partial(violations, full_train_loader)
    additional_losses = (
        ("val loss", val_loss, False),
        ("test loss", test_loss, False),
        ("val R^2", val_r_squared, False),
        ("test R^2", test_r_squared, False),
        ("val violations", val_viols, False),
        ("test violations", test_viols, False),
    )

    if args.optim == "Adam":
        optimizer_class = torch.optim.Adam
    elif args.optim == "RMSprop":
        optimizer_class = torch.optim.RMSprop
    else:  # SGD
        optimizer_class = torch.optim.SGD

    info(f"Starting training ({args.restarts} restarts)")
    best_val_loss = torch.inf
    best_network = None
    for restart_i in tqdm(range(args.restarts), desc="Training Restarts"):
        network = new_network()
        # train in two stages: continue training with a smaller learning rate
        # once the first stage has finished.

        # first stage
        optimizer = optimizer_class(network.parameters(), lr=args.lr1)
        training_loop = TrainingLoop(
            network, optimizer, train_loss
        )

        loss_logger = LogLoss(
            log_frequency=100, average_training_loss=True,
            additional_losses=additional_losses
        )
        training_loop.add_post_iteration_hook(loss_logger)
        if args.show_plots:
            tensorboard_dir = (
                ".tensorboard" + os.sep + experiment_name + os.sep + str(restart_i)
            )
            tensorboard_writer = tensorboard.writer.SummaryWriter(log_dir=tensorboard_dir)
            training_loop.add_post_iteration_hook(TensorboardLossPlot(
                tensorboard_writer, frequency=10,
                additional_losses=additional_losses,
            ))

        training_loop.add_termination_criterion(IterationMaximum(15000))
        training_loop.add_termination_criterion(ValidationSet(
            val_loss,
            iterations_between_validations=10,
            acceptable_increase_length=5,
            tolerance_fraction=0.01,  # 1% increase/decrease
            reset_parameters=True,
        ))
        training_loop.add_termination_criterion(TrainingLossChange(
            change_threshold=1.0, iteration_block_size=5, num_blocks=5,
        ))
        training_loop.execute()

        # second stage
        info("Second training stage.")
        optimizer = optimizer_class(network.parameters(), lr=args.lr2)
        training_loop = TrainingLoop(
            network, optimizer, train_loss
        )

        training_loop.add_post_iteration_hook(loss_logger)
        if args.show_plots:
            training_loop.add_post_iteration_hook(TensorboardLossPlot(
                tensorboard_writer, frequency=10,
                training_loss_tag='training loss stage 2',
                additional_losses=[
                    (name + " stage 2", loss_fn, average)
                    for name, loss_fn, average in additional_losses
                ],
            ))

        training_loop.add_termination_criterion(IterationMaximum(15000))
        training_loop.add_termination_criterion(ValidationSet(
            val_loss,
            iterations_between_validations=10,
            acceptable_increase_length=3,
            tolerance_fraction=0.01,  # 1% increase/decrease
            reset_parameters=True,
        ))
        training_loop.add_termination_criterion(TrainingLossChange(
            change_threshold=0.1, iteration_block_size=5, num_blocks=5,
        ))
        training_loop.execute()

        val_loss_value = val_loss()
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_network = network
    info("Training finished.")

    info("Evaluating best network.")
    network = best_network

    grid_points = 5
    grid_dose = torch.linspace(dose_min, dose_max, grid_points)
    grid_Cl = torch.linspace(Cl_min, Cl_max, grid_points)
    grid_V = torch.linspace(V_min, V_max, grid_points)
    grid_ka = torch.linspace(ka_min, ka_max, grid_points)
    grid_t = torch.linspace(0.0, args.t_max, args.samples_per_patient)
    grid = torch.cartesian_prod(grid_dose, grid_Cl, grid_V, grid_ka, grid_t)
    grid = TensorDataset(
        grid,
        Dataset1CMPPO.compute_output(*grid.T).unsqueeze(-1)
    )
    grid_loader = DataLoader(grid, batch_size=len(grid))

    grid_loss = partial(loss, grid_loader)
    grid_r_squared = partial(r_squared, grid_loader)
    grid_viols = partial(violations, grid_loader)

    best_net_train_loss = full_train_loss().item()
    best_net_val_loss = val_loss().item()
    best_net_test_loss = test_loss().item()
    best_net_grid_loss = grid_loss().item()
    best_net_train_r_squared = full_train_r_squared().item()
    best_net_val_r_squared = val_r_squared().item()
    best_net_test_r_squared = test_r_squared().item()
    best_net_grid_r_squared = grid_r_squared().item()
    best_net_train_viols = full_train_viols().item()
    best_net_val_viols = val_viols().item()
    best_net_test_viols = test_viols().item()
    best_net_grid_viols = grid_viols().item()

    info(
        f"Training results: \n\n"
        f"                    loss    R^2  violations (%)\n"
        f" training set: {best_net_train_loss:9.4f},  {best_net_train_r_squared:.2f}, {best_net_train_viols:14.2f}\n"
        f" val set:      {best_net_val_loss:9.4f},  {best_net_val_r_squared:.2f}, {best_net_val_viols:14.2f}\n"
        f" test set:     {best_net_test_loss:9.4f},  {best_net_test_r_squared:.2f}, {best_net_test_viols:14.2f}\n"
        f" grid:         {best_net_grid_loss:9.4f},  {best_net_grid_r_squared:.2f}, {best_net_grid_viols:14.2f}\n"
    )

    network_path = Path("output_1_CMP_PO", f"{output_name}_network.pyt")
    torch.save(network, network_path)
    stats_and_info = {
        "dataset": {
            "N": args.N,
            "samples_per_patient": args.samples_per_patient,
            "Cl": args.Cl,
            "V": args.V,
            "ka": args.ka,
            "WT": args.WT,
            "dose": args.dose,
            "t_max": args.t_max,
            "training_set_size": len(train_set),
            "validation_set_size": len(val_set),
            "test_set_size": len(test_set),
            "path": str(dataset.dataset_path),
        },
        "network": {
            "architecture": args.architecture,
            "batch_size": args.batch_size,
            "lr1": args.lr1,
            "lr2": args.lr2,
            "optim": args.optim,
            "restarts": args.restarts,
            "trained_network_path": str(network_path),
        },
        "results": {
            "training_set_loss": best_net_train_loss,
            "validation_set_loss": best_net_val_loss,
            "test_set_loss": best_net_test_loss,
            "grid_loss": best_net_grid_loss,
            "training_set_r_squared": best_net_train_r_squared,
            "validation_set_r_squared": best_net_val_r_squared,
            "test_set_r_squared": best_net_test_r_squared,
            "grid_r_squared": best_net_grid_r_squared,
            "training_set_violations": best_net_train_viols,
            "validation_set_violations": best_net_val_viols,
            "test_set_violations": best_net_test_viols,
            "grid_violations": best_net_grid_viols,
        },
    }
    with open(Path("output_1_CMP_PO", f"{output_name}_info.yaml"), "wt") as file:
        yml = yaml.YAML(typ="safe")
        yml.Representer = yaml.RoundTripRepresenter
        yml.indent = 4
        yml.sequence_dash_offset = 0
        yml.default_flow_style = False
        yml.dump(stats_and_info, file)
    info(f"Stored trained network in {network_path}.")
