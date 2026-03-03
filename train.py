import argparse
import random
import time

import numpy as np
import torch
from utils import *
from collections import OrderedDict
from sklearn import preprocessing
from model import *
import scanpy
import os

if __name__ == "__main__":

    # lam1 = ['Quake_10x_Tongue','Baron_human', 'Baron_human', 'Quake_10x_Spleen', 'Quake_10x_Trachea','Quake_Smart-seq2_Limb_Muscle','Quake_10x_Mammary_Gland']
    # lam2 = ['Quake_Smart-seq2_Tongue','Muraro', 'Enge', 'Quake_Smart-seq2_Spleen', 'Quake_Smart-seq2_Trachea','Quake_Smart-seq2_Limb_Muscle','Quake_Smart-seq2_Mammary_Gland']

    lam1 = ['Baron_human']
    lam2 = ['Muraro']
    aa = [10]
    bb = [100]
    cc = [1]
    dd = [0.1]
    for i in range(len(lam1)):
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default:1e-3')
        parser.add_argument('--beta', type=float, default=aa[i], help='β for CE loss, default:1e-3')
        parser.add_argument('--gamma', type=float, default=bb[i], help='γ for contrastive loss, default:1e-3')
        parser.add_argument('--delta', type=float, default=cc[i], help='δ for graph reconstruction loss, default:1e-3')
        parser.add_argument('--lambda_', type=float, default=dd[i], help='λ for DEC loss, default:1e-3')
        parser.add_argument('--phi', type=float, default=0.5, help='phi, default:1e-3')
        parser.add_argument('--dims', default=[1000, 256, 64, 32, 256], type=int,
                            help='The number of neurons of the encoder')
        parser.add_argument('--training_epoch', type=int, default=100,
                            help='epoch of train stage, default:200')
        parser.add_argument('--clustering_epoch', type=int, default=100,
                            help='epoch of clustering stage, default:100')
        parser.add_argument('--device', type=str, default="GPU",
                            help='use GPU, or else use cpu (setting as "False")')

        parser.add_argument('--sample_size', type=float, default=1.,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--debias', type=float, default=0.,
                            help='Debias factor for contrastive learning.')
        # parser.add_argument('--node', type=int, default=20, help='do data augmentation.')
        parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'], default='Kmeans',
                            help='clustering algorithm (default: Kmeans)')
        parser.add_argument('--verbose', action='store_true', help='chatty')
        parser.add_argument('--data_aug', type=int, default=1, help='do data augmentation.')
        parser.add_argument('--neg_type', type=float, default=0, help='0,selection;1 not selection')
        parser.add_argument('--pos_type', type=float, default=0, help='0,selection;1 not selection')

        args = parser.parse_args()

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(args.device)
        # args.device = "cpu"
        random_seed = 8888
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)  # 设置所有CUDA设备的PyTorch随机数生成器的种子
            torch.backends.cudnn.deterministic = True  # 设置CuDNN（CUDA深度神经网络库）为确定性模式。
            torch.backends.cudnn.benchmark = False  # 禁用了CuDNN的自动调优（benchmark）功能

        datapath1 = './Data/real/'+ lam1[i] + '/' + lam1[i] + '.h5ad'
        datapath2 = './Data/real/'+ lam2[i] + '/' + lam2[i] + '.h5ad'
        # datapath1 = "./Data/real/Baron_human/Baron_human.h5ad"
        # datapath2 = "./Data/real/Enge/Enge.h5ad"
        # datapath2 = "./Data/real/Muraro/Muraro.h5ad"
        # datapath1 = "./Data/real/Quake_10x_Tongue/Quake_10x_Tongue.h5ad"
        # datapath2 = "./Data/real/Quake_Smart-seq2_Tongue/Quake_Smart-seq2_Tongue.h5ad"
        X1, Y1, genes_name1 = read_real2(datapath1)
        X2, Y2, genes_name2 = read_real2(datapath2)
        X1 = pd.DataFrame(X1, columns=genes_name1)
        X2 = pd.DataFrame(X2, columns=genes_name2)

        # 找出两个数据中共同的基因名
        common_gene_names = np.intersect1d(genes_name1, genes_name2)
        # common_genes = list(set(genes_name1) & set(genes_name2))

        # 从data1中选取基因名为common_gene_names的数据
        X1 = X1.loc[:, common_gene_names]

        # 从data2中选取基因名为common_gene_names的数据
        X2 = X2.loc[:, common_gene_names]
        Y = np.concatenate((Y1, Y2), axis=0)
        X = np.concatenate((X1, X2), axis=0)

        batch_label = np.ones(Y.shape)
        batch_label[:Y1.size] = 0

        X = X.astype(np.float32)

        adata, rawData, dataset = preprocess(X, Y, batch_label, highly_genes=args.dims[0])
        celltype = adata.obs['celltype']
        cell_type, celltype = np.unique(celltype, return_inverse=True)
        size_factor = adata.obs['size_factors'].values
        Zscore_data = preprocessing.scale(dataset)
        n_clusters = cell_type.shape[0]
        init_model = scLGGCL(dims=args.dims, n_clusters=n_clusters, args=args)
        init_model.to(args.device)

        start = time.time()
        _, z = init_model.pretrain(Zscore_data, rawData, size_factor, batch_label, celltype, args)

        adata_z = scanpy.AnnData(z)  # scanpy中常见的数据结构是AnnData，它是一个用于存储数据的对象，

        adj, r_adj = adata_knn(adata_z, method='gauss', knn=False,
                               n_neighbors=15, metric='cosine')

        pred_label, _, _ = init_model.clustering(Zscore_data, adj, r_adj, rawData, celltype, n_clusters,
                                                 batch_label,
                                                 size_factor, args)
        end = time.time()

        ari = adjusted_rand_score(celltype[batch_label == 1], pred_label[batch_label == 1])
        nmi = normalized_mutual_info_score(celltype[batch_label == 1], pred_label[batch_label == 1])
        acc = np.round(accuracy_score(celltype[batch_label == 1], pred_label[batch_label == 1]), 4)

        ARI = ari
        NMI = nmi
        ACC = acc
        # print("Final ARI %.3f, NMI %.3f, ACC %.3f" % (ari, nmi, acc))
        print("Final ARI %.3f, NMI %.3f, ACC %.3f, Times: %f" % (ari, nmi, acc, end - start))

        file_path = './results/time.csv'
        # file_path = './results/Baron_Enge_Muraro.csv'
        # file_path = './results/smaller_imbalance_' + str(a) + '.csv'
        # 创建一个 DataFrame
        data = pd.DataFrame(
            {'A': lam2[i], 'Times': [end - start], 'ARI': [ari], 'NMI': [nmi], 'ACC': [acc]})
        # # 检查文件是否存在
        if os.path.exists(file_path):
            # 如果文件存在，追加数据，不写入列名和索引
            data.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # 如果文件不存在，创建文件并写入数据，包括列名，不写入索引
            data.to_csv(file_path, mode='w', header=True, index=False)