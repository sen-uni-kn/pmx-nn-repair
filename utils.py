import itertools
import re
from typing import Callable

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deep_opt import NeuralNetwork


_network_architecture_rx = re.compile(r"(\d+,)*\d+,?")
_distribution_rx = re.compile(
    r"(?P<distr>(log-normal)|(uniform))\[(?P<params>(\d+(\.\d*)?,)*\d+(\.\d*)?,?)]"
)


def sample_data(size: int, distribution_desc: str, rng: torch.Generator) -> torch.Tensor:
    """
    Sample :code:`size` values from the distribution described by
    :code:`distribution_desc`.

    :param size: How many values to sample.
    :param distribution_desc: The description of the random distribution to sample
        from.
        This method supports these formats:
         - :code:`log-normal[ln_mean,std]`, and
         - :code:`uniform[offset,spread]`.
        For :code:`log-normal[ln_mean,std]`, the data is sampled from a log-normal
        distribution with standard deviation :code:`std` and mean
        :math:`\ln(ln\_mean)`.
        Both :code`std` and :code:`ln_mean` need to be floating point values
        that can be converted by :code:`float`.
        For :code:`uniform[offset,spread]`, the data is sampled from a uniform
        distribution over the range :math:`[offset,offset+spread]`.
        Similarly to :code:`log-normal`, both :code:`offset` and :code:`spread`
        need to be floating point values.
    :param rng: The random number generator to use for sampling.
    :return: A tensor with shape :code:`(size, 1)` containing samples of the
        random distribution.
    """
    distribution_desc = _distribution_rx.fullmatch(distribution_desc)
    if distribution_desc is None:
        raise ValueError(f"Distribution is not specified correctly.")
    parameters = distribution_desc.group("params").split(",")
    parameters = [param.strip() for param in parameters if len(param) > 0]
    try:
        parameters = [float(param) for param in parameters]
    except ValueError as ex:
        raise ValueError(
            f"Found non-float parameter in distribution description"
        ) from ex
    if distribution_desc.group("distr") == "log-normal":
        if len(parameters) != 2:
            raise ValueError(
                f"Log-normal distribution requires a mean and a standard deviation. "
                f"Got {len(parameters)} values instead."
            )
        ln_mean, std = parameters
        return ln_mean * torch.exp(std * torch.randn(size, 1, generator=rng))
    elif distribution_desc.group("distr") == "uniform":
        if len(parameters) != 2:
            raise ValueError(
                f"Uniform distribution requires an offset and a spread. "
                f"Got {len(parameters)} values instead."
            )
        offset, spread = parameters
        return torch.rand(size, 1, generator=rng) * spread + offset
    else:
        raise ValueError(f"Unknown distribution: {distribution_desc.group('distr')}")


def get_network_factory(
    architecture: str, data_in: torch.Tensor, data_out: torch.Tensor,
    input_mins: torch.Tensor, input_maxs: torch.Tensor,
) -> Callable[[], NeuralNetwork]:
    """
    Creates a function that creates new randomly initialized neural networks
    with the given architecture.

    :param architecture: The network architecture as a comma-separated list of
        hidden layer sizes.
    :param data_in: A dataset of network inputs.
        Used for calculating normalization parameters.
    :param data_out: A dataset of network outputs.
        Used for calculating normalization parameters.
    :return: A function without arguments that returns new randomly initialized
        neural networks.
    """
    num_inputs = data_in.size(1)
    num_outputs = data_out.size(1)
    network_architecture = [int(size.strip()) for size in architecture.split(',')]
    network_architecture.append(num_outputs)
    layer_in_sizes = [num_inputs] + network_architecture[:-1]

    def new_network():
        layers = itertools.chain(*[
            (nn.Linear(num_in, num_out), nn.ReLU())
            for num_in, num_out in zip(layer_in_sizes, network_architecture)
        ])
        layers = list(layers)[:-1]  # don't need terminal ReLU
        return NeuralNetwork(
            mins=input_mins,
            maxes=input_maxs,
            means_inputs=data_in.mean(dim=0),
            ranges_inputs=data_in.std(dim=0),
            means_outputs=data_out.mean(dim=0),
            ranges_outputs=data_out.std(dim=0),
            modules=layers,
            inputs_shape=(num_inputs,),
            outputs_shape=(num_outputs,)
        )
    return new_network


class ReduceLROnPlateau2:
    """
    A wrapper of :code:`torch.optim.lr_scheduler.ReduceLROnPlateau`
    that independently calculates the metric for checking for plateaus.
    """
    def __init__(self, metric_fn, *args, **kwargs):
        self.base_scheduler = ReduceLROnPlateau(*args, **kwargs)
        self.metric_fn = metric_fn

    def step(self):
        metric_value = self.metric_fn()
        self.base_scheduler.step(metric_value)
