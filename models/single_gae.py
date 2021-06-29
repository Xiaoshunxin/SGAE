"""
    重构损失
    解码器是一个线性变换
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops_al import dot_product_decode
from layers.graph_conv import GraphConvolution


class Single_GAE(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """
            the simplest gae framework with a gcn encoder and a linear decoder
        :param input_dim: the number dimension of input feature
        :param hidden_dim: the output dimension of GCN and the input dimension of the linear
        """
        super(Single_GAE, self).__init__()
        self.encoder_gcn = GraphConvolution(input_dim, hidden_dim)
        self.decoder_linear = nn.Linear(hidden_dim, input_dim)

    def copy_weights(self, encoder: GraphConvolution, decoder: nn.Linear) -> None:
        """
        Utility method to copy the weights of self into the given encoder and decoder, where
        encoder should be instances of GraphConvolution and decoder is nn.Linear.
        :param encoder: encoder Linear unit
        :param decoder: decoder Linear unit
        :return: None
        """
        encoder.weight.data.copy_(self.encoder_gcn.weight)
        encoder.bias.data.copy_(self.encoder_gcn.bias)
        decoder.weight.data.copy_(self.decoder_linear.weight)
        decoder.bias.data.copy_(self.decoder_linear.bias)

    def encode(self, fea: torch.Tensor, adj: torch.sparse) -> torch.Tensor:
        """
            the computing of encoder
        :param fea: input feature matrix
        :param adj: adjacency matrix
        :return:  the hidden matrix of input feature matrix
        """
        hidden = self.embedding = self.encoder_gcn(fea, adj)
        return hidden

    def decode(self, fea: torch.Tensor):
        """
            the computing of decoder
        :param fea: input feature matrix
        :return:
        """
        output = self.decoder_linear(fea)
        return output

    def forward(self,
                fea: torch.Tensor,
                adj: torch.sparse):
        """
            the computing of graph convolution
        :param fea: input feature matrix
        :param adj: adjacency matrix
        :return:  the reconstruct matrix of input feature matrix
        """
        return self.decode(self.encode(fea, adj))
