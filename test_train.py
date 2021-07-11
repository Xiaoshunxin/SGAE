import os
import time
import torch
import random
import argparse
import numpy as np
from torch.optim import RMSprop
from sklearn.cluster import KMeans

import utils.ops_model as ae
from utils.ops_ev import get_evaluation_results
from utils.ops_io import load_data, save_training_process
from models.stacked_gae_clk import StackedGraphAutoencoder


if __name__ == '__main__':
    # Configuration settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=str, default='0', help='The number of cuda device.')
    parser.add_argument('--n_repeated', type=int, default=5, help='Number of repeated experiments')
    parser.add_argument('--seed', type=int, default=1009, help='Number of seed.')

    parser.add_argument('--direction', type=str, default='./data/datasets/mat',
                        help='The direction of the datasets')
    parser.add_argument('--dataset_name', type=str, default='CNAE', help='The dataset used for training/testing')
    parser.add_argument('--load_saved', action='store_true', default=True, help='Whether to load adjacency matrix.')
    parser.add_argument('--normalization', action='store_true', default=True, help='Whether to use the node feature')
    parser.add_argument('--normalization_type', type=str, default='normalize', help='default normalize')
    parser.add_argument('--k_nearest_neighbors', type=int, default=50, help='the number of nearest neighbors')
    parser.add_argument('--prunning_one', action='store_true', default=True, help='Whether to use prunning one')
    parser.add_argument('--prunning_two', action='store_true', default=True, help='Whether to use prunning two')
    parser.add_argument('--common_neighbors', type=int, default=2, help='threshold of common neighbors')

    parser.add_argument('--dimension', type=int, default=256, help='the number of hidden dimension')

    parser.add_argument('--layerwise_weights_path', type=str, default='./data/layerwise_weights/', help='weights path')
    parser.add_argument('--show_patience', type=int, default=100, help='the number of show patience')
    parser.add_argument('--loss_patience', type=int, default=100, help='the number of loss patience')
    parser.add_argument('--save_patience', type=int, default=200001,
                        help='the frequency about saving the training result in the finetune stage')
    parser.add_argument('--layerwise_epochs', type=int, default=20000, help='number of layer-wise training epochs.')
    parser.add_argument('--layerwise_lr', type=float, default=0.0001, help='learning rate of layer-wise training.')
    parser.add_argument('--layerwise_momentum', type=float, default=0.9, help='number of layer-wise momentum.')
    parser.add_argument('--layerwise_weight_decay', type=float, default=0.0, help='number of layer-wise weight decay.')
    parser.add_argument('--finetune_epochs', type=int, default=20000, help='number of finetunning epochs.')
    parser.add_argument('--finetune_lr', type=float, default=0.00001, help='learning rate of finetunning.')
    parser.add_argument('--finetune_momentum', type=float, default=0.9, help='the number of momentum.')
    parser.add_argument('--finetune_weight_decay', type=float, default=0.0, help='number of finetunning weight decay.')
    parser.add_argument('--finetune_lk_weight', type=float, default=0.005, help='weight of link prediction loss')

    parser.add_argument('--n_init', type=int, default=5, help='Number of different initialization for K-Means.')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel worker for K-Means.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    all_ACC = []
    all_NMI = []
    all_Purity = []
    all_ARI = []
    all_Layer_Wise_Time = []
    all_Finetune_Time = []
    all_Clustering_Time = []
    all_Total_Time = []

    for i in range(args.n_repeated):
        if i == 0:
            args.load_saved = False
            # args.load_saved = True
        else:
            args.load_saved = True

        features, labels, adj_wave, adj_hat, norm, weight_tensor = \
            load_data(direction_path=args.direction, dataset_name=args.dataset_name, normalization=args.normalization,
                      normalization_type=args.normalization_type, load_saved=args.load_saved,
                      k_nearest_neighobrs=args.k_nearest_neighbors, prunning_one=args.prunning_one,
                      prunning_two=args.prunning_two, common_neighbors=args.common_neighbors)
        args.input_dim = features.shape[1]
        args.num_classes = len(np.unique(labels))

        autoencoder = StackedGraphAutoencoder([args.input_dim, 256, 64, 16], final_activation=None)

        if args.cuda:
            autoencoder.cuda()
            # features = features.cuda(non_blocking=True)
            features = features.cuda()
            adj_hat = adj_hat.cuda()
            adj_wave = adj_wave.cuda()
            weight_tensor = weight_tensor.cuda()

        start_time = time.time()
        # layer-wise training the autoencoder or load the trained one.
        layerwise_weights_path = args.layerwise_weights_path + args.dataset_name + '.pkl'
        if args.load_saved is not True:
            print('layer-wise training stage...')
            layerwise_start_time = time.time()
            ae.layerwise(fea=features, adj_hat=adj_hat, autoencoder=autoencoder, num_epochs=args.layerwise_epochs,
                         show_patience=args.show_patience, max_loss_patience=args.loss_patience, labels=labels, cuda=args.cuda,
                         # scheduler=lambda x: StepLR(x, 20000, gamma=0.1),
                         optimizer=lambda model: RMSprop(model.parameters(), lr=args.layerwise_lr,
                                                         weight_decay=args.layerwise_weight_decay,
                                                         momentum=args.layerwise_momentum))
            layerwise_cost_time = time.time() - layerwise_start_time
            all_Layer_Wise_Time.append(layerwise_cost_time)

            # save the layer-wise trained weights
            torch.save(autoencoder.state_dict(), layerwise_weights_path)
        else:
            print("loading the layer-wise trained wieghts....")
            autoencoder.load_state_dict(torch.load(layerwise_weights_path))
            if args.cuda:
                autoencoder.cuda()

        print('finetuning stage.')
        finetune_start_time = time.time()
        ae_optimizer = RMSprop(params=autoencoder.parameters(), lr=args.finetune_lr,
                               momentum=args.finetune_momentum, weight_decay=args.finetune_weight_decay)
        List_ID, List_loss, List_ACC, List_NMI, List_ARI = \
            ae.finetune(fea=features, labels=labels, adj_hat=adj_hat, autoencoder=autoencoder, optimizer=ae_optimizer,
                        # scheduler=StepLR(ae_optimizer, 20000, gamma=0.1), corruption=0.2,
                        num_epochs=args.finetune_epochs, max_loss_patience=args.loss_patience,
                        show_patience=args.show_patience, lk_weight=args.finetune_lk_weight,
                        norm=norm, adj_wave=adj_wave, weight_tensor=weight_tensor, save_patience=args.save_patience)
        finetune_cost_time = time.time() - finetune_start_time
        # save_training_process(args.dataset_name, i, List_ID, List_loss, List_ACC, List_NMI, List_ARI)

        print('k-Means stage')
        clustering_start_time = time.time()
        kmeans = KMeans(n_clusters=args.num_classes, n_init=args.n_init, n_jobs=args.n_jobs)
        autoencoder.eval()
        embeddings = autoencoder.encode(features, adj_hat).detach().cpu()
        predicted = kmeans.fit_predict(embeddings.numpy())
        clustering_cost_time = time.time() - clustering_start_time

        print('Evaluation stage')
        ACC, NMI, Purity, ARI = get_evaluation_results(labels.numpy(), predicted)
        print(ACC, NMI, Purity, ARI)
        total_cost_time = time.time() - start_time

        all_ACC.append(ACC)
        all_NMI.append(NMI)
        all_Purity.append(Purity)
        all_ARI.append(ARI)
        all_Finetune_Time.append(finetune_cost_time)
        all_Clustering_Time.append(clustering_cost_time)
        all_Total_Time.append(total_cost_time)

    # append result to .txt file
    fp = open("results_" + args.dataset_name + ".txt", "a+", encoding="utf-8")
    # fp = open("results.txt", "a+", encoding="utf-8")
    fp.write("Dataset: {}\n".format(args.dataset_name))
    fp.write("finetune_lk_weight: {}\n".format(args.finetune_lk_weight))
    # fp.write("common_neighbors: {}\n".format(args.common_neighbors))
    fp.write("ACC: {:.2f}\t{:.2f}\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    fp.write("NMI: {:.2f}\t{:.2f}\n".format(np.mean(all_NMI) * 100, np.std(all_NMI) * 100))
    fp.write("Purity: {:.2f}\t{:.2f}\n".format(np.mean(all_Purity) * 100, np.std(all_Purity) * 100))
    fp.write("ARI: {:.2f}\t{:.2f}\n".format(np.mean(all_ARI) * 100, np.std(all_ARI) * 100))
    fp.write("Layer-wise Time: {:.2f}\t{:.2f}\n".format(np.mean(all_Layer_Wise_Time), np.std(all_Layer_Wise_Time)))
    fp.write("Finetune Time: {:.2f}\t{:.2f}\n".format(np.mean(all_Finetune_Time), np.std(all_Finetune_Time)))
    fp.write("Clustering Time: {:.2f}\t{:.2f}\n".format(np.mean(all_Clustering_Time), np.std(all_Clustering_Time)))
    fp.write("Total Time: {:.2f}\t{:.2f}\n\n".format(np.mean(all_Total_Time), np.std(all_Total_Time)))
    fp.close()
