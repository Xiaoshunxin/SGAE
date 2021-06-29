import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from typing import Iterable, Optional, List
from cytoolz.itertoolz import sliding_window

from layers.graph_conv import GraphConvolution


def build_encoder_units(dimensions: Iterable[int], activation: Optional[nn.Module]) -> List[nn.Module]:
    """
    Given a list of dimensions and optional activation, return a list of units where each unit is a gcn encoder
    layer followed by an activation layer.
    :param dimensions: iterable of dimensions for the chain
    :param activation: activation layer to use e.g. nn.ReLU, set to None to disable
    :return: list of instances of Sequential
    """
    def single_unit(input_dim: int, output_dim: int) -> nn.Module:
        unit = [('gcn', GraphConvolution(input_dim, output_dim))]
        if activation is not None:
            unit.append(('activation', activation))
        return nn.Sequential(OrderedDict(unit))
    return [single_unit(input_dim, output_dim) for input_dim, output_dim in sliding_window(2, dimensions)]


def build_decoder_units(dimensions: Iterable[int], activation: Optional[nn.Module]) -> List[nn.Module]:
    """
    Given a list of dimensions and optional activation, return a list of units where each unit is a linear decoder
    layer followed by an activation layer.
    :param dimensions: iterable of dimensions for the chain
    :param activation: activation layer to use e.g. nn.ReLU, set to None to disable
    :return: list of instances of Sequential
    """
    def single_unit(input_dim: int, output_dim: int) -> nn.Module:
        unit = [('linear', nn.Linear(input_dim, output_dim))]
        if activation is not None:
            unit.append(('activation', activation))
        return nn.Sequential(OrderedDict(unit))
    return [single_unit(input_dim, output_dim) for input_dim, output_dim in sliding_window(2, dimensions)]


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred