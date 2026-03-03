import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import time
# from torch_geometric.nn import GCNConv, GATConv, TAGConv
from loss import *
from utils import *
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
import math
import copy
import clustering

# A random Gaussian noise uses in the ZINB-based denoising autoencoder.
class GaussianNoise(nn.Module):
    def __init__(self, device, sigma=1, is_relative_detach=True):
        super(GaussianNoise,self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, device = device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x

#用于创建一个包含多个深拷贝模块的 nn.ModuleList，其作用是生成 N 个完全独立的相同 PyTorch 模块 module 的副本。
def clones(module, N): #每次深拷贝生成的模块是相互独立的，权重和参数不共享。
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GraphConstructor(nn.Module):
    def __init__(self, input_dim, h, phi, device, dropout=0):
        super(GraphConstructor, self).__init__()
        assert input_dim % h == 0 #确保输入维度可以被头数整除，便于分头计算。

        self.d_k = input_dim // h  ## 每个头的特征维度
        self.h = h  #多头机制的头数。
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.h), 2)  #包含两个线性变换（query 和 key），用于对输入进行维度变换。
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)  #一个线性层，用于将多头注意力的结果整合成单个分数。
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)#一个可学习参数，用作邻接矩阵的阈值。

        self.device = device

    def forward(self, query, key):
        query, key = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]  #对输入的 query 和 key 进行线性变换，并重塑为多头的格式：

        attns = self.attention(query.squeeze(2), key.squeeze(2))
        adj = torch.where(attns >= self.phi, torch.ones(attns.shape).to(self.device), torch.zeros(attns.shape).to(self.device))#若注意力分数 attns 大于等于阈值 phi，则连接两个节点（值为 1）。否则不连接（值为 0）。

        return adj

    def attention(self, query, key):
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)) \
                 / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn

class ADF(nn.Module):
    def __init__(self, n_adf, h):
        super(ADF, self).__init__()
        self.w1 = nn.Linear(n_adf, h)

    def forward(self, adf_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(adf_in)), dim=1)

        return weight_output

# Three different activation function uses in the ZINB-based denoising autoencoder.
MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e4)
DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e3)
PiAct = lambda x: 1/(1+torch.exp(-1 * x))


def cal_centers(r, label, class_number):
    centers = torch.zeros((class_number, r.size(1)), device=r.device)  # 使用训练数据集的聚类中心并放置在相同设备上
    for i in range(class_number):
        centers[i] = torch.mean(r[label == i], dim=0)  # 求平均值
    return centers

# A dot product operation uses in the decoder of GAE.
def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def LGCL(features, output, args, kk:int=1):
    device = args.device
    sample_size = args.sample_size
    pos_type = args.pos_type
    neg_type = args.neg_type
    # sample nodes
    node_mask = torch.empty(features.shape[0], dtype=torch.float32).uniform_(0, 1).to(device)
    node_mask = node_mask < sample_size  # 'True' means that this node is selected to participate in training

    # positive selection
    if pos_type == 0:
        y_pre = output.detach()
        y_pre_p = y_pre[node_mask].to(device)
        _, y_topind = torch.topk(y_pre_p, kk)
        y_ol = torch.zeros(y_pre_p.shape).to(device)
        y_ol = y_ol.scatter_(1, y_topind, 1)
        out_pos_mask = torch.mm(y_ol, y_ol.T) > 0  # 'True' means that the node pair has the same pseudo-label
        out_pos_mask = out_pos_mask.to(device)
        del y_ol, y_topind
        torch.cuda.empty_cache()
    else:
        out_pos_mask = torch.eye(node_mask.sum()).to(device)

    # clustering
    # pos_mask = out_pos_mask
    deepcluster = clustering.__dict__[args.clustering](y_pre.shape[1])
    clustering_index = deepcluster.cluster(features, verbose=args.verbose)
    clustering_index = torch.LongTensor(clustering_index).to(device)
    clustering_index_np = torch.zeros(features.shape[0], y_pre.shape[1]).to(device)
    clustering_index_np = clustering_index_np.scatter_(1, clustering_index, 1)
    clustering_index_np = clustering_index_np[node_mask]
    clustering_neg_mask = torch.mm(clustering_index_np, clustering_index_np.T) <= 0
    clustering_true_mask = torch.mm(clustering_index_np,
                                    clustering_index_np.T) == 1  # 'True' indicates that the node pair belongs to the same cluster
    clustering_true_mask = clustering_true_mask.to(device)
    clustering_neg_mask = clustering_neg_mask.to(device)

    # self-checking mechanism
    pos_mask = torch.mul(out_pos_mask,
                         clustering_true_mask).to(device)  # 'True' indicates that the node pair has the same pseudo-label and belongs to the same cluster
    gl = torch.mul(out_pos_mask, clustering_true_mask).to(device)
    flb = torch.sum(out_pos_mask == 1)
    fla = torch.sum(gl == 1)
    # sccs = flb - fla
    # sccs.detach().cpu().numpy()
    # sccs = sccs.tolist()
    # fp = open('512/filter.txt', 'r+')
    # n = fp.read()
    # fp.write(str(sccs) + ",")

    # negative selection
    if neg_type == 0:
        y_pre_n = y_pre[node_mask].to(device)
        _, y_poslabel = torch.topk(y_pre_n, kk)
        y_pl = torch.zeros(y_pre_n.shape).to(device)
        y_pl = y_pl.scatter_(1, y_poslabel, 1)
        out_neg_mask = torch.mm(y_pl, y_pl.T) <= 0  # 'True' means that the node pair has different pseudo-label
        out_neg_mask = out_neg_mask.to(device)
        del y_pl, y_poslabel
        torch.cuda.empty_cache()
    else:
        out_neg_mask = (1 - torch.eye(node_mask.sum())).to(device)

    # # reweighting negative ndoes
    # neg_mask = out_neg_mask
    y_soft_pre = F.softmax(y_pre[node_mask], dim=1).to(device)
    y_pre_mean = torch.mean(y_soft_pre, dim=1, keepdim=True).to(device)
    y_pre_std = torch.std(y_soft_pre, dim=1, keepdim=True).to(device)
    neg_weight = torch.zeros(y_soft_pre.shape).to(device)
    for i in range(neg_weight.shape[0]):
        neg_weight[i] = torch.add(y_soft_pre[i], torch.neg(y_pre_mean[i])) * torch.add(y_soft_pre[i],
                                                                                       torch.neg(y_pre_mean[i]))
        neg_weight[i] = torch.neg(neg_weight[i] / 2 * (y_pre_std[i]))
        neg_weight[i] = torch.exp(neg_weight[i])
    neg_mask = out_neg_mask
    neg_mask = neg_mask.to(torch.float64)
    for i in range(neg_mask.shape[0]):
        neg_label_ind = torch.max(y_pre[node_mask], dim=1, keepdim=True)[1].T.to(device)
        out = torch.take(neg_weight[i], neg_label_ind).to(device)
        neg_mask[i] = neg_mask[i] * out

    return pos_mask, neg_mask, node_mask


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        if active:
            output = torch.tanh(output)
        return output

# A multi-head attention layer has two different input (query and key/value).
class AttentionWide(nn.Module):
    def __init__(self, emb, p = 0.2, heads=8, mask=False):
        super().__init__()

        self.emb = emb
        self.heads = heads
        # self.mask = mask
        self.dropout = nn.Dropout(p)
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, y):
        b = 1
        t, e = x.size()
        h = self.heads
        # assert e == self.emb, f'Input embedding dimension {{e}} should match layer embedding dim {{self.emb}}'

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.dropout(self.toqueries(y)).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention

        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        # if self.mask:
        #     mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        self.attention_weights = dot
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)


class scLGGCL(nn.Module):
    def __init__(self, dims, n_clusters, args, heads=5, sigma=0.1, dropout=0.4):
        super(scLGGCL, self).__init__()
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.dims = dims
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, dims[3]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.Gnoise = GaussianNoise(sigma=0.1, device=args.device)
        self.enc_1 = nn.Linear(dims[0], dims[1])
        self.BN_1 = nn.BatchNorm1d(dims[1])
        self.enc_2 = nn.Linear(dims[1], dims[2])
        self.BN_2 = nn.BatchNorm1d(dims[2])
        self.z_layer = nn.Linear(dims[2], dims[3])
        #
        # self.gcn1 = GCNConv(dims[0], dims[1])
        # self.gcn2 = GCNConv(dims[1], dims[2])
        # self.gcn3 = GCNConv(dims[2], dims[3])
        #
        self.gnn_0 = GNNLayer(dims[0], dims[1])
        self.gnn_1 = GNNLayer(dims[1], dims[2])
        self.gnn_2 = GNNLayer(dims[2], dims[3])
        # self.gnn_0 = GCNConv(dims[0], dims[1])
        # self.gnn_1 = GCNConv(dims[1], dims[2])
        # self.gnn_2 = GCNConv(dims[2], dims[3])

        self.attn0 = AttentionWide(dims[1], heads=heads)
        self.attn1 = AttentionWide(dims[2], heads=heads)
        self.attn2 = AttentionWide(dims[3], heads=heads)

        self.out_y = nn.Linear(dims[3], self.n_clusters)

        self.dec_2 = nn.Linear(dims[3], dims[-1])

        self.calcu_pi = nn.Linear(dims[-1], dims[0])
        self.calcu_disp = nn.Linear(dims[-1], dims[0])
        self.calcu_mean = nn.Linear(dims[-1], dims[0])

        self.zinb = ZINBLoss(theta_shape=(dims[0],)).to(args.device)
        self.re_loss = nn.MSELoss().to(args.device)
        self.loss_ce = nn.CrossEntropyLoss().to(args.device)
        self.KL_loss = DECLoss().to(args.device)
        # self.con = contrastiveLoss().to(args.device)
        # self.pair_loss = PairLoss().to(args.device)

    def forward(self, x, adj):
        # adj = adj - torch.diag_embed(adj.diag())
        # edge_index = torch.nonzero(adj == 1).T

        enc_h1 = self.BN_1(F.relu(self.enc_1(self.Gnoise(x))))
        h0 = self.gnn_0(x, adj)
        # enc_h1 = (self.attn0(enc_h1, h0)).squeeze(0) + enc_h1
        h0 = (self.attn0(h0, enc_h1)).squeeze(0) + h0
        # enc_h1 = (enc_h1 + h0) / 2.0 + enc_h1
        # h0 = (enc_h1 + h0) / 2.0 + h0
        h1 = self.gnn_1(h0, adj)
        enc_h2 = self.BN_2(F.relu(self.enc_2(self.Gnoise(enc_h1))))
        h1 = (self.attn1(h1, enc_h2)).squeeze(0) + h1
        # enc_h2 = (enc_h2 + h1) / 2.0 + enc_h2
        # h1 = (enc_h2 + h1) / 2.0 + h1
        h2 = self.gnn_2(h1, adj)

        # enc_h2 = (self.attn1(enc_h2, h1)).squeeze(0)+enc_h2
        z = self.z_layer(self.Gnoise(enc_h2))
        # h2 = self.gnn_2((self.attn1(enc_h2, h1)).squeeze(0), adj)
        z = (self.attn2(h2, z)).squeeze(0) + h2
        # z = (h2 + z) / 2.0 + z
        # z = (h2 + z) / 2.0 + h2
        # z = (self.attn2(z, h2)).squeeze(0) + z

        y_hat = F.softmax(self.out_y(z), dim=1)  # 训练使用带噪声的z,预测使用不带噪声的z
        # decoder
        A_pred = dot_product_decode(z)
        # A_pred = dot_product_decode(z)
        # dec_h1 = F.relu(self.dec_1(z))
        # dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h2 = F.relu(self.dec_2(z))

        pi = PiAct(self.calcu_pi(dec_h2))
        mean = MeanAct(self.calcu_mean(dec_h2))
        disp = DispAct(self.calcu_disp(dec_h2))
        return z, A_pred, pi, mean, disp, y_hat

    def forward1(self, x):
        enc_h1 = self.BN_1(F.relu(self.enc_1(self.Gnoise(x))))
        enc_h2 = self.BN_2(F.relu(self.enc_2(self.Gnoise(enc_h1))))
        # enc_h1 = F.relu(self.enc_1(self.Gnoise(x)))
        # enc_h2 = F.relu(self.enc_2(self.Gnoise(enc_h1)))
        z = self.z_layer(enc_h2)
        # dec_h1 = F.relu(self.dec_1(z))
        # dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h2 = F.relu(self.dec_2(z))

        pi = PiAct(self.calcu_pi(dec_h2))
        mean = MeanAct(self.calcu_mean(dec_h2))
        disp = DispAct(self.calcu_disp(dec_h2))
        return z, pi, mean, disp
    def suplabel_lossv6neg(self, z1: torch.Tensor, z2: torch.Tensor, mask: torch.Tensor, neg_mask: torch.Tensor,
                           pos_mask: torch.Tensor, debias, mean_type: int = 1, tau: int = 1):
        if mean_type == 0:
            s_value = torch.mm(z1, z1.t())
            b_value = torch.mm(z1, z2.t())
            s_value_max, _ = torch.max(s_value, dim=1, keepdim=True)
            s_value = s_value - s_value_max.detach()
            b_value_max, _ = torch.max(b_value, dim=1, keepdim=True)
            b_value = b_value - b_value_max.detach()
            s_value = torch.exp(s_value / tau)
            b_value = torch.exp(b_value / tau)
        else:
            s_value = torch.exp(torch.mm(z1, z1.t()) / tau)
            b_value = torch.exp(torch.mm(z1, z2.t()) / tau)
        # import ipdb;ipdb.set_trace()
        # value_zi = b_value.diag().unsqueeze(0).T
        value_zi = (s_value + b_value) * pos_mask.float()
        value_zi = value_zi.sum(dim=1, keepdim=True)

        # import ipdb;ipdb.set_trace()
        value_neg = (s_value + b_value) * neg_mask.float()
        value_neg = value_neg.sum(dim=1, keepdim=True)
        neg_sum = 2 * neg_mask.sum(dim=1, keepdim=True)
        value_neg = (value_neg - value_zi * neg_sum * debias) / (1 - debias)
        value_neg = torch.max(value_neg, neg_sum * math.exp(-1.0 / tau))
        value_mu = value_zi + value_neg

        # import ipdb;ipdb.set_trace()
        loss = -torch.log(value_zi / value_mu)
        return loss

    def cl_lossaug(self, z1: torch.Tensor, z2: torch.Tensor, mask: torch.Tensor, train_mask: torch.Tensor, labels,
                   neg_mask, pos_mask,
                   train_type, att_type, debias, neg: int = 1, mean: bool = True):
        # neg   train8.py=1  train11.py=0
        # h1 = self.projection(z1)
        # h2 = self.projection(z2)
        h1 = z1
        h2 = z2
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        # import ipdb;ipdb.set_trace()
        if train_type == 0:
            # labels = labels[train_mask]
            h1 = h1[train_mask]
            h2 = h2[train_mask]
            if neg == 0:
                neg_mask = neg_mask[train_mask].T
                neg_mask = neg_mask[train_mask].T
            # neg_sample = torch.empty(neg_mask.shape,dtype=torch.float32).uniform_(0,1).cuda()
            # neg_sample = torch.where(neg_sample<0.857,1,0)
            # neg_mask = neg_mask * neg_sample
            # import ipdb;ipdb.set_trace()
        if att_type == 0:
            pass
        else:
            loss1 = self.suplabel_lossv6neg(h1, h2, mask, neg_mask, pos_mask, debias)
            loss2 = self.suplabel_lossv6neg(h2, h1, mask, neg_mask, pos_mask, debias)
            ret = (loss1 + loss2) / 2

        ret = ret.mean() if mean else ret.sum()
        return ret


    def pretrain(self, Zscore_data, rawData, size_factor, batch_label, celltype, args):
        start_time = time.time()
        data, batch, celltype1, rawData = [torch.Tensor(var).to(args.device) for var in
                                  [Zscore_data, batch_label, celltype, rawData]]
        sf = torch.autograd.Variable((torch.from_numpy(size_factor[:, None]).type(torch.FloatTensor)).to(args.device),
                                     requires_grad=True)  # 将大小因子 size_factor 转换为 PyTorch 张量并设置为可训练变量
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=args.lr, amsgrad=True)

        for epoch in range(args.training_epoch):
            z, pi, mean, disp = self.forward1(data)
            # z, A_pred, pi, mean, disp, y_hat = self.forward(data, adj)

            zinb_loss = self.zinb(mean * sf, pi, target=rawData, theta=disp)
            # con_loss = self.con(mean * sf, rawData)

            loss = zinb_loss
            # loss = loss.item()
            # loss = zinb_loss

            # if (epoch + 1) % 2 == 0:
            #     print("epoch %d, loss %.4f, zinb_loss %.4f"
            #           % (epoch + 1, loss, zinb_loss))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        elapsed_time = time.time() - start_time
        with torch.no_grad():
            z, pi, mean, disp = self.forward1(data)
        z = z.detach().cpu().numpy()#Tongue_latent

        # print("Finish Training! Elapsed time: {:.4f} seconds".format(elapsed_time))
        return elapsed_time, z


    def clustering(self, Zscore_data, adj, r_adj, rawData, celltype, n_clusters, batch_label, size_factor, args):
        start_time = time.time()

        data, adj, r_adj, batch, celltype1, rawData = [torch.Tensor(var).to(args.device) for var in [Zscore_data, adj, r_adj, batch_label, celltype, rawData]]
        celltype1 = celltype1.long()
        sf = torch.autograd.Variable((torch.from_numpy(size_factor[:, None]).type(torch.FloatTensor)).to(args.device), requires_grad=True)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=args.lr, amsgrad=True)


        for epoch in range(args.clustering_epoch):
            z, A_pred, pi, mean, disp, y_hat = self.forward(data, adj)

            label = dist_2_label(y_hat)
            ce = self.loss_ce(y_hat[batch==0], celltype1[batch==0])
            zinb_loss = self.zinb(mean * sf, pi, target=rawData, theta=disp)
            # re_graphloss = self.re_loss(A_pred.view(-1), adj.to_dense().view(-1))
            re_graphloss = self.re_loss(A_pred.view(-1), r_adj.view(-1))

            pos_mask, neg_mask, node_mask = LGCL(data, y_hat, args)

            if args.data_aug == 1:
                features1 = drop_feature(data, 0.3)
                features2 = drop_feature(data, 0.4)
                _, _, _, _, _, out1 = self.forward(features1, adj)
                _, _, _, _, _, out2 = self.forward(features2, adj)
                del features1, features2
                torch.cuda.empty_cache()
                loss_cl = self.cl_lossaug(out1, out2, None, node_mask, celltype1, neg_mask, pos_mask, 0, 1, args.debias)
            else:
                loss_cl = torch.tensor(0.0).to(args.device)

            # 修改这里：使用 args.beta (CE loss), args.gamma (contrastive loss), args.delta (graph reconstruction loss)
            loss = zinb_loss + args.delta * re_graphloss + args.beta * ce + args.gamma * loss_cl
            # loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()

            # if epoch % 2 == 0:
            #     acc = accuracy_score(celltype[batch_label==1], label[batch_label==1])
            #     ari = adjusted_rand_score(celltype[batch_label==1], label[batch_label==1])
            #
            #     print("#Epoch %3d: acc: %.4f,ARI: %.4f, ZINB Loss: %.4f CE_loss: %.4f, re_loss: %.4f, loss_cl: %.4f" % (
            #         epoch, acc, ari, zinb_loss, ce, re_graphloss, loss_cl))

        for epoch in range(args.clustering_epoch):
            z, A_pred, pi, mean, disp, y_hat = self.forward(data, adj)
            label = dist_2_label(y_hat)
            if epoch == 0:
                latent1 = np.nan_to_num(z.detach().cpu().numpy())
                kmeans = KMeans(n_clusters, init="k-means++", n_init=10)
                kmeans_pred = kmeans.fit_predict(latent1)
                last_label = kmeans_pred
                y_pp = torch.Tensor(label).long().to(args.device)
                y_pp[batch == 0] = celltype1[batch == 0]
                # cluster_centers = cal_centers(z[batch == 0], celltype1[batch == 0], n_clusters)
                cluster_centers = cal_centers(z, y_pp, n_clusters)

                with torch.no_grad():
                    self.cluster_layer.copy_(cluster_centers)

            ce = self.loss_ce(y_hat[batch == 0], celltype1[batch == 0])
            zinb_loss = self.zinb(mean * sf, pi, target=rawData, theta=disp)
            re_graphloss = self.re_loss(A_pred.view(-1), r_adj.view(-1))
            # re_graphloss = self.re_loss(A_pred.view(-1), adj.to_dense().view(-1))
            kl_loss, p = self.KL_loss(z, self.cluster_layer)
            # p_loss = self.pair_loss(celltype1, batch, y_hat)

            pos_mask, neg_mask, node_mask = LGCL(data, y_hat, args)
            if args.data_aug == 1:
                features1 = drop_feature(data, 0.3)
                features2 = drop_feature(data, 0.4)
                _, _, _, _, _, out1 = self.forward(features1, adj)
                _, _, _, _, _, out2 = self.forward(features2, adj)
                del features1, features2
                torch.cuda.empty_cache()
                loss_cl = self.cl_lossaug(out1, out2, None, node_mask, celltype1, neg_mask, pos_mask, 0, 1, args.debias)
            else:
                loss_cl = torch.tensor(0.0).to(args.device)

            # 修改这里：使用 args.lambda_ (DEC loss), args.beta (CE loss), args.gamma (contrastive loss), args.delta (graph reconstruction loss)
            loss = zinb_loss + args.delta * re_graphloss + args.beta * ce + args.lambda_ * kl_loss + args.gamma * loss_cl
            label = dist_2_label(y_hat)
            # loss = loss.item()

            if epoch % 2 == 0:
                # acc = accuracy_score(celltype[batch_label == 1], label[batch_label == 1])
                # ari = adjusted_rand_score(celltype[batch_label == 1], label[batch_label == 1])
                # print(
                #     "#Epoch %3d: acc: %.4f,ARI: %.4f, ZINB Loss: %.4f CE_loss: %.4f, re_loss: %.4f, D_loss: %.4f, loss_cl: %.4f" % (
                #         epoch, acc, ari, zinb_loss, ce, re_graphloss, kl_loss, loss_cl))
                if np.sum(label != last_label) / len(last_label) < 0.001:
                    break
                else:
                    last_label = label
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()

        elapsed_time = time.time() - start_time
        #
        # print("Finish Training! Elapsed time: {:.4f} seconds".format(elapsed_time))

        return last_label, self.cluster_layer, elapsed_time