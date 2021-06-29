import pdb
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from utils.ops_al import build_encoder_units, build_decoder_units, dot_product_decode


class StackedGraphAutoencoder(nn.Module):
    
    def __init__(self,
                 dimensions: List[int],
                 activation: nn.Module = nn.ReLU(),
                 final_activation: Optional[nn.Module] = nn.ReLU()):
        """
            Stacked Denoising GAE
        :param dimensions: list of dimensions occurring in a single stack
        :param activation: activation layer to use for all but final activation, default torch.nn.ReLU
        :param final_activation: final activation layer to use, set to None to disable, default torch.nn.ReLU
        """
        super(StackedGraphAutoencoder, self).__init__()
        self.dimensions = dimensions
        self.input_dim = dimensions[0]
        self.hidden_dim = dimensions[-1]

        # construct the encoder
        encoder_units = build_encoder_units(dimensions[:-1], activation)
        encoder_units.extend(build_encoder_units([dimensions[-2], dimensions[-1]], None))
        self.encoder = nn.Sequential(*encoder_units)

        # construct the decoder
        decoder_units = build_decoder_units(reversed(dimensions[1:]), activation)
        decoder_units.extend(build_decoder_units([dimensions[1], dimensions[0]], final_activation))
        self.decoder = nn.Sequential(*decoder_units)

    def get_stack(self, index: int) -> Tuple[nn.Module, nn.Module]:
        """
        Given an index which is in [0, len(self.dimensions) - 2] return the corresponding subautoencoder
        for layer-wise pretraining.
        :param index: subautoencoder index
        :return: tuple of encoder and decoder units
        """
        if (index > len(self.dimensions) - 2) or (index < 0):
            raise ValueError('Requested subautoencoder cannot be constructed, index out of range.')
        return self.encoder[index].gcn, self.decoder[-(index + 1)].linear

    def encode(self, fea: torch.Tensor, adj: torch.sparse) -> torch.Tensor:
        output = fea
        for seq in self.encoder:
            if len(seq) == 1:
                output = seq[0](output, adj)
            elif len(seq) == 2:
                output = seq[0](output, adj)
                output = seq[1](output)
        return output

    def decode(self, fea: torch.Tensor) -> torch.Tensor:
        output = fea
        for seq in self.decoder:
            if len(seq) == 1:
                output = seq[0](output)
            elif len(seq) == 2:
                output = seq[0](output)
                output = seq[1](output)
        return output

    def forward(self, fea: torch.Tensor, adj: torch.sparse):
        """
            the computing of SGAE
        :param fea: input feature matrix
        :param adj: adjacency matrix
        :return: the reconstruct matrix of input feature matrix
        """
        encoded = self.encode(fea, adj)
        return self.decode(encoded), dot_product_decode(encoded)
