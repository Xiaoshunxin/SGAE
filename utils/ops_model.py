import pdb
import time
import numpy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from typing import Callable, Optional, Any

from models.single_gae import Single_GAE
from models.stacked_gae_clk import StackedGraphAutoencoder
from utils.ops_ev import get_evaluation_results


def layerwise(fea: torch.Tensor,
             adj_hat: torch.sparse,
             autoencoder: StackedGraphAutoencoder,
             optimizer: Callable[[nn.Module], torch.optim.Optimizer],
             labels: torch.Tensor,
             num_epochs: int = 50000,
             scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
             cuda: bool = True,
             max_loss_patience: Optional[int] = 100,
             show_patience: int = 100) -> None:
    """
    :param fea: input feature matrix
    :param adj_hat: normalized adjacency matrix
    :param autoencoder: instance of an SGAE to train
    :param num_epochs: number of training epochs
    :param optimizer: function taking model and returning optimizer
    :param scheduler: function taking optimizer and returning scheduler, or None to disable
    :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
    :param cuda: whether CUDA is used, defaults to True
    :param max_loss_patience: the maximum number to stop the training by the loss
    :param show_patience: the frequency to show the traning detail
    :return:
    """
    current_fea = fea
    num_sub_ae = len(autoencoder.dimensions) - 1
    for index in range(num_sub_ae):
        print('layer-wise training the num_ae: ', index)
        encoder, decoder = autoencoder.get_stack(index)
        input_dim = autoencoder.dimensions[index]
        hidden_dim = autoencoder.dimensions[index + 1]
        sub_autoencoder = Single_GAE(input_dim, hidden_dim)
        if cuda:
            sub_autoencoder = sub_autoencoder.cuda()
        ae_optimizer = optimizer(sub_autoencoder)
        ae_scheduler = scheduler(ae_optimizer) if scheduler is not None else scheduler
        layerwise_train(fea=current_fea, adj_hat=adj_hat, autoencoder=sub_autoencoder, corruption=None,  # already have dropout in the DAE
              num_epochs=num_epochs, optimizer=ae_optimizer, scheduler=ae_scheduler,
              max_loss_patience=max_loss_patience, show_patience=show_patience)
        # copy the weights
        sub_autoencoder.copy_weights(encoder, decoder)

        kmeans = KMeans(n_clusters=10, n_init=5, n_jobs=8)
        predicted = kmeans.fit_predict(sub_autoencoder.embedding.detach().type_as(labels).numpy())
        ACC, NMI, Purity, ARI = get_evaluation_results(labels.numpy(), predicted)
        print("the performance of the num_ae: ", index)
        print(ACC, NMI, Purity, ARI)

        # pass the dataset through the encoder part of the subautoencoder
        if index != (num_sub_ae - 1):
            current_fea = sub_autoencoder.embedding.detach().type_as(labels).numpy()
            if cuda:
                current_fea = torch.Tensor(current_fea).cuda()
            else:
                current_fea = torch.Tensor(current_fea)
        else:
            current_fea = None


def layerwise_train(fea: torch.Tensor,
                    adj_hat: torch.sparse,
                    autoencoder: nn.Module,
                    num_epochs: int,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Any = None,
                    corruption: Optional[float] = None,
                    max_loss_patience: Optional[int] = 100,
                    show_patience: int = 100) -> None:
    """
    :param fea: input feature matrix
    :param adj_hat: normalized adjacency matrix
    :param autoencoder: instance of an SGAE or DGAE to train
    :param num_epochs: number of training epochs
    :param optimizer: function taking model and returning optimizer
    :param scheduler: function taking optimizer and returning scheduler, or None to disable
    :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
    :param max_loss_patience: the maximum number to stop the training by the loss
    :param show_patience: the frequency to show the traning detail
    :return:
    """
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    loss_patience = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        autoencoder.train()
        if scheduler is not None:
            scheduler.step()

        if corruption is not None:
            output = autoencoder(F.dropout(fea, corruption), adj_hat)
        else:
            output = autoencoder(fea, adj_hat)

        optimizer.zero_grad()
        loss = loss_function(output, fea)
        loss.backward()
        optimizer.step(closure=None)

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch + 1
            loss_patience = 0
        else:
            loss_patience += 1

        if loss_patience > max_loss_patience:
            print("Break by loss patience!")
            print("Best Epoch:", '%04d' % (best_epoch), "best loss=", "{:.5f}".format(best_loss))
            break
        if (epoch + 1) % show_patience == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_value))


def finetune(fea: torch.Tensor,
             labels: torch.Tensor,
             adj_hat: torch.sparse,
             adj_wave,
             norm,
             weight_tensor,
             autoencoder: nn.Module,
             num_epochs: int,
             optimizer: torch.optim.Optimizer,
             scheduler: Any = None,
             corruption: Optional[float] = None,
             max_loss_patience: Optional[int] = 100,
             show_patience: int = 100,
             save_patience: int = 20000,
             lk_weight: float = 1.0):
    """
    :param fea: input feature matrix
    :param adj: normalized adjacency matrix
    :param autoencoder: instance of an SGAE or DGAE to train
    :param num_epochs: number of training epochs
    :param optimizer: function taking model and returning optimizer
    :param scheduler: function taking optimizer and returning scheduler, or None to disable
    :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
    :param max_loss_patience: the maximum number to stop the training by the loss
    :param show_patience: the frequency to show the traning detail
    :return:
    """
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    loss_patience = 0
    best_epoch = 0

    List_ID = []
    List_loss = []
    List_ACC = []
    List_NMI = []
    List_ARI = []
    num_classes = len(np.unique(labels))

    start_time = time.time()
    used_epochs = 0
    for epoch in range(num_epochs):
        used_epochs += 1
        autoencoder.train()

        # if scheduler is not None:
        #     scheduler.step()
        # if corruption is not None:
        #     output = autoencoder(F.dropout(fea, corruption), adj)
        # else:
        #     output = autoencoder(fea, adj)
        Z_pred, A_pred = autoencoder(fea, adj_hat)
        optimizer.zero_grad()
        loss = lk_weight * norm * F.binary_cross_entropy(A_pred.view(-1), adj_wave.to_dense().view(-1),
                                                              weight=weight_tensor)
        loss += loss_function(Z_pred, fea)
        loss.backward()
        optimizer.step(closure=None)

        loss_value = float(loss.item())

        if (epoch+1) % save_patience == 0:
            List_ID.append(epoch+1)
            List_loss.append(loss_value)
            with torch.no_grad():
                embeddings = autoencoder.encode(fea, adj_hat).detach().cpu()
                kmeans = KMeans(n_clusters=num_classes, n_init=5, n_jobs=8)
                predicted = kmeans.fit_predict(embeddings.numpy())
                ACC, NMI, Purity, ARI = get_evaluation_results(labels.numpy(), predicted)
                List_ACC.append(ACC)
                List_NMI.append(NMI)
                List_ARI.append(ARI)
                print(ACC, NMI, ARI)

        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch + 1
            loss_patience = 0
        else:
            loss_patience += 1

        if loss_patience > max_loss_patience:
            print("Break by loss patience!")
            print("Best Epoch:", '%04d' % (best_epoch), "best loss=", "{:.5f}".format(best_loss))
            break
        if (epoch + 1) % show_patience == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_value))

    time_cost = time.time() - start_time
    print("####################################")
    print("time_cost: ", time_cost)
    print("used_epochs: ", used_epochs)
    print("per_epoch_time: ", time_cost/used_epochs)
    return List_ID, List_loss, List_ACC, List_NMI, List_ARI
