import argparse
import sys
from datetime import datetime
import logging
from logging import info
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils import tensorboard
import ruamel.yaml as yaml

from datasets.dataset_1_cmp_po import Dataset1CMPPO

from nn_repair import repair_network, RepairStatus
from deep_opt import Property, BoxConstraint
from nn_repair.training import (
    TrainingLoop, LogLoss,
    TrainingLossChange, IterationMaximum,
    ValidationSet, TensorboardLossPlot,
    Divergence, ResetOptimizer,
)
from nn_repair.backends import PenaltyFunction, PenaltyFunctionRepairDelegate
from nn_repair.verifiers import ERAN

from experiments.experiment_base import seed_rngs, TrackingLossFunction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Repair 1 CMP PO model.')

    parser.add_argument(
        "train_stats_and_info",
        help="A path to a stats and info file from training neural networks. "
             "This file is used to load the dataset and obtain the trained network "
             "to repair."
    )

    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--batch_size", type=int, default=None,
        help="The mini batch size used for training the network. "
             "Defaults to the batch size used for training the network. "
    )
    training_group.add_argument(
        "--lr", type=float, default=None,
        help="The learning rate for repair. "
             "Defaults to the second learning rate (lr2) for training "
             "the network."
    )
    training_group.add_argument(
        "--optim", choices=["Adam", "RMSprop", "SGD", "default"],
        default="default",
        help="The training algorithm to use for training. "
             "Defaults to the training algorithm used for training."
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
        help="Show some plots of repair progress."
    )

    args = parser.parse_args()
    with open(args.train_stats_and_info, "rt") as file:
        train_info = yaml.safe_load(file)
    if args.batch_size is None:
        args.batch_size = train_info["network"]["batch_size"]
    if args.lr is None:
        args.lr = train_info["network"]["lr2"]
    if args.optim == "default":
        args.optim = train_info["network"]["optim"]
    dataset_info = train_info["dataset"]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = (
        f"repair_1_cmp_po_"
        f"N_{dataset_info['N']}_"
        f"samples_per_patient_{dataset_info['samples_per_patient']}_"
        f"Cl_{dataset_info['Cl']}_V_{dataset_info['V']}_ka_{dataset_info['ka']}_"
        f"WT_{dataset_info['WT']}_dose_{dataset_info['dose']}_"
        f"t_max_{dataset_info['t_max']}_"
        f"batch_size_{args.batch_size}_"
        f"lr_{args.lr}_optim_{args.optim}_"
        f"{timestamp}"
    )
    output_name = timestamp if args.output_name is None else args.output_name
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )

    seed_rngs(546113353761942)

    network_path = train_info["network"]["trained_network_path"]
    info(f"Loading network from {network_path}")
    network = torch.load(network_path)

    info("Loading Dataset...")
    dataset = Dataset1CMPPO(
        root=Path("output_1_CMP_PO"),
        size=dataset_info['N'] * dataset_info['samples_per_patient'],
        samples_per_patient=dataset_info['samples_per_patient'],
        Cl_distribution=dataset_info['Cl'],
        V_distribution=dataset_info['V'],
        ka_distribution=dataset_info['ka'],
        WT_distribution=dataset_info['WT'],
        dose_at_unit_WT=dataset_info['dose'],
        t_max=dataset_info['t_max'],
    )

    train_end = dataset_info["training_set_size"]
    val_end = train_end + dataset_info["validation_set_size"]

    train_set = Subset(dataset, indices=list(range(train_end)))
    val_set = Subset(dataset, indices=list(range(train_end, val_end)))
    test_set = Subset(dataset, indices=list(range(val_end, len(dataset))))
    assert len(train_set) == dataset_info["training_set_size"]
    assert len(val_set) == dataset_info["validation_set_size"]
    assert len(test_set) == dataset_info["test_set_size"]

    batch_size = args.batch_size
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    full_train_loader = DataLoader(train_set, batch_size=len(train_set))
    val_loader = DataLoader(val_set, batch_size=len(val_set))
    val_loader_batch = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=len(test_set))

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

    grid_points = 5
    dose_min, Cl_min, V_min, ka_min, t_min = network.mins
    dose_max, Cl_max, V_max, ka_max, t_max = network.maxes
    grid_dose = torch.linspace(dose_min, dose_max, grid_points)
    grid_Cl = torch.linspace(Cl_min, Cl_max, grid_points)
    grid_V = torch.linspace(V_min, V_max, grid_points)
    grid_ka = torch.linspace(ka_min, ka_max, grid_points)
    grid_t = torch.linspace(t_min, t_max, dataset_info["samples_per_patient"])
    grid = torch.cartesian_prod(grid_dose, grid_Cl, grid_V, grid_ka, grid_t)
    grid = TensorDataset(
        grid,
        Dataset1CMPPO.compute_output(*grid.T).unsqueeze(-1)
    )
    grid_loader = DataLoader(grid, batch_size=len(grid))

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
    grid_loss = partial(loss, grid_loader)
    grid_r_squared = partial(r_squared, grid_loader)
    grid_viols = partial(violations, grid_loader)
    additional_losses = (
        ("val loss", val_loss, False),
        ("test loss", test_loss, False),
        ("val R^2", val_r_squared, False),
        ("test R^2", test_r_squared, False),
        ("val violations", val_viols, False),
        ("test violations", test_viols, False),
    )

    wrapped_loss = TrackingLossFunction(train_loss, "task loss")
    additional_losses = wrapped_loss.get_additional_losses() + additional_losses
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    elif args.optim == "RMSprop":
        optimizer = torch.optim.RMSprop(network.parameters(), lr=args.lr)
    else:  # SGD
        optimizer = torch.optim.SGD(network.parameters(), lr=args.lr)
    training_loop = TrainingLoop(network, optimizer, wrapped_loss)
    training_loop.add_pre_training_hook(ResetOptimizer(optimizer))
    wrapped_loss.register_loss_resetting_hook(training_loop)

    loss_logger = LogLoss(
        log_frequency=100, average_training_loss=True,
        additional_losses=additional_losses
    )
    training_loop.add_post_iteration_hook(loss_logger)
    if args.show_plots:
        tensorboard_dir = (
            str(Path(".tensorboard", experiment_name))
        )
        tensorboard_writer = tensorboard.writer.SummaryWriter(log_dir=tensorboard_dir)
        training_loop.add_post_iteration_hook(TensorboardLossPlot(
            tensorboard_writer, frequency=10,
            additional_losses=additional_losses,
        ))

    training_loop.add_termination_criterion(IterationMaximum(2500))
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
    # stop training if something goes wrong
    training_loop.add_termination_criterion(Divergence(network.parameters()))

    cx_remover = PenaltyFunctionRepairDelegate(
        training_loop, penalty_function=PenaltyFunction.L1, maximum_updates=100
    )

    spec = [
        Property(
            lower_bounds={},
            upper_bounds={},
            output_constraint=BoxConstraint(0, '>=', 0),
            property_name=f"no negative outputs"
        )
    ]

    initial_net_train_loss = full_train_loss().item()
    initial_net_val_loss = val_loss().item()
    initial_net_test_loss = test_loss().item()
    initial_net_grid_loss = grid_loss().item()
    initial_net_train_r_squared = full_train_r_squared().item()
    initial_net_val_r_squared = val_r_squared().item()
    initial_net_test_r_squared = test_r_squared().item()
    initial_net_grid_r_squared = grid_r_squared().item()
    initial_net_train_viols = full_train_viols().item()
    initial_net_val_viols = val_viols().item()
    initial_net_test_viols = test_viols().item()
    initial_net_grid_viols = grid_viols().item()

    repair_status, _ = repair_network(
        network,
        spec,
        cx_remover,
        verifier=ERAN(use_acasxu_style=False, exit_mode="early_exit")
    )

    repaired_net_train_loss = full_train_loss().item()
    repaired_net_val_loss = val_loss().item()
    repaired_net_test_loss = test_loss().item()
    repaired_net_grid_loss = grid_loss().item()
    repaired_net_train_r_squared = full_train_r_squared().item()
    repaired_net_val_r_squared = val_r_squared().item()
    repaired_net_test_r_squared = test_r_squared().item()
    repaired_net_grid_r_squared = grid_r_squared().item()
    repaired_net_train_viols = full_train_viols().item()
    repaired_net_val_viols = val_viols().item()
    repaired_net_test_viols = test_viols().item()
    repaired_net_grid_viols = grid_viols().item()

    if repair_status is not RepairStatus.SUCCESS:
        info(f"Repair failed with status {repair_status}")
        sys.exit()

    info("Repair successful.")
    info(
        f"Repair results (values in brackets are before repair):\n\n"
        f"                    loss                R^2         violations (%)\n"
        f" training set: {repaired_net_train_loss:9.4f} "
        f"({initial_net_train_loss:9.4f}),  "
        f"{repaired_net_train_r_squared:.2f} ({initial_net_train_r_squared:.2f}), "
        f"{repaired_net_train_viols:5.2f} ({initial_net_train_viols:5.2f})\n"
        f" val set:      {repaired_net_val_loss:9.4f} "
        f"({initial_net_val_loss:9.4f}),  "
        f"{repaired_net_val_r_squared:.2f} ({initial_net_val_r_squared:.2f}), "
        f"{repaired_net_val_viols:5.2f} ({initial_net_val_viols:5.2f})\n"
        f" test set:     {repaired_net_test_loss:9.4f} "
        f"({initial_net_test_loss:9.4f}),  "
        f"{repaired_net_test_r_squared:.2f} ({initial_net_test_r_squared:.2f}), "
        f"{repaired_net_test_viols:5.2f} ({initial_net_test_viols:5.2f})\n"
        f" grid set:     {repaired_net_grid_loss:9.4f} "
        f"({initial_net_grid_loss:9.4f}),  "
        f"{repaired_net_grid_r_squared:.2f} ({initial_net_grid_r_squared:.2f}), "
        f"{repaired_net_grid_viols:5.2f} ({initial_net_grid_viols:5.2f})\n"
    )
    network_path = Path("output_1_CMP_PO", f"{output_name}_network.pyt")
    torch.save(network, network_path)
    stats_and_info = {
        "training_info": args.train_stats_and_info,
        "network": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optim": args.optim,
            "repaired_network_path": str(network_path),
        },
        "results": {
            "training_set_loss": repaired_net_train_loss,
            "validation_set_loss": repaired_net_val_loss,
            "test_set_loss": repaired_net_test_loss,
            "grid_loss": repaired_net_grid_loss,
            "training_set_r_squared": repaired_net_train_r_squared,
            "validation_set_r_squared": repaired_net_val_r_squared,
            "test_set_r_squared": repaired_net_test_r_squared,
            "grid_r_squared": repaired_net_grid_r_squared,
            "training_set_violations": repaired_net_train_viols,
            "validation_set_violations": repaired_net_val_viols,
            "test_set_violations": repaired_net_test_viols,
            "grid_violations": repaired_net_grid_viols,
        },
    }
    with open(Path("output_1_CMP_PO", f"{output_name}_info.yaml"), "wt") as file:
        yml = yaml.YAML(typ="safe")
        yml.Representer = yaml.RoundTripRepresenter
        yml.indent = 4
        yml.sequence_dash_offset = 0
        yml.default_flow_style = False
        yml.dump(stats_and_info, file)
    info(f"Stored repaired network in {network_path}.")
