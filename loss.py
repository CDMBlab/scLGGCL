import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from scipy.special import digamma
import torch.nn.functional as F
import numpy as np

class contrastiveLoss(nn.Module):
    def __init__(self, temperature=1.):
        super(contrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * z_i.size(0)#
        z = torch.cat((z_i, z_j), dim=0)
        self.batch_size = z_i.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        z = nn.functional.normalize(z, p=2, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss = loss/N   #平均每个样本,平均每个像素
        return loss


class Lgamma(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return torch.lgamma(input)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors

        res = Variable(torch.from_numpy(digamma(input.cpu().numpy())).type_as(input))
        return grad_output*res

class ZINBLoss(torch.nn.Module):
    def __init__(self, theta_shape=None, pi_ridge=0.0):
        super().__init__()
        self.pi_ridge = pi_ridge

        if theta_shape is not None:
            theta = torch.Tensor(*theta_shape).log_normal_(0.1, 0.01)
            self.register_parameter('theta', torch.nn.Parameter(theta))

    def forward(self, mean, pi, target, theta=None, _epsilon = 1e-6):
        eps = _epsilon

        if theta is None:
            theta = 1.0 / (torch.exp(self.theta).clamp(max=1e6) + eps)

        # reuse existing NB nll
        nb_case = self.nb(mean, target, theta) - torch.log(1.0 - pi + eps)

        # print("pi:{},\n nb_case:{}".format(pi.detach().numpy(),nb_case.detach().numpy()))

        zero_nb = torch.pow(theta / (theta + mean + eps), theta)
        # print("zero_nb:{}".format(zero_nb.detach().numpy()))

        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        # print("zero_case:{}".format(zero_case.detach().numpy()))

        zero_mask = (target == 0.0).type_as(zero_case)
        # print("zero_mask:{}".format(zero_mask.detach().numpy()))
        nb_mask = 1.0 - zero_mask
        # print("nb_mask:{}".format(nb_mask.detach().numpy()))

        result = zero_mask * zero_case + nb_mask * nb_case

        if self.pi_ridge:
            ridge = self.pi_ridge * pi.pow(2)
            result += ridge

        return result.mean()

    def nb(self, input, target, theta, _epsilon = 1e-6):
        # input is mean
        lgamma = Lgamma.apply
        eps = _epsilon

        t1 = -lgamma(target + theta + eps)
        t2 = lgamma(theta + eps)
        t3 = lgamma(target + 1.0)
        t4 = -(theta * (torch.log(theta + eps)))
        t5 = -(target * (torch.log(input + eps)))
        t6 = (theta + target) * torch.log(theta + input + eps)

        res = t1 + t2 + t3 + t4 + t5 + t6
        return res

class DECLoss(torch.nn.Module):
    def __init__(self):
        super(DECLoss, self).__init__()  # super() 函数是用于调用父类(超类)的一个方法

    def forward(self, hidden, cluster, alpha=1.0):  # gamma: weight of clustering loss
        # 计算距离矩阵
        dist = torch.sum((hidden.unsqueeze(1) - cluster) ** 2, dim=2)

        # 计算q分布
        q = 1.0 / (1.0 + dist / alpha) ** ((alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)

        # 计算p分布
        f = q ** 2 / torch.sum(q, dim=0)
        p = f / torch.sum(f, dim=1, keepdim=True)

        # 计算交叉熵损失
        q_clipped = torch.clamp(q, min=1e-10, max=1.0)
        p_clipped = torch.clamp(p, min=1e-10, max=1.0)

        cross_entropy = -p * torch.log(q_clipped) - (1 - p) * torch.log(1 - q_clipped)
        # cross_entropy = torch.sum(cross_entropy)
        cross_entropy = torch.mean(torch.sum(cross_entropy, dim=1))

        return cross_entropy, p


class PairLoss(torch.nn.Module):
    def __init__(self):
        super(PairLoss, self).__init__()  # super() 函数是用于调用父类(超类)的一个方法

    # def forward(self, label_vec, mask_vec, y_pred):  # gamma: weight of clustering loss
    #     # 创建标签矩阵
    #     label_mat = label_vec.view(-1, 1) - label_vec.view(1, -1)
    #     label_mat = (label_mat == 0).float()
    #
    #     # 创建掩码矩阵
    #     mask_mat = mask_vec.view(-1, 1) * mask_vec.view(1, -1)
    #
    #     # 归一化
    #     normalize_discriminate = F.normalize(y_pred, p=2, dim=1)
    #
    #     # 计算相似度
    #     similarity = torch.matmul(normalize_discriminate, normalize_discriminate.t())
    #
    #     # 计算交叉熵
    #     cross_entropy = mask_mat * (-label_mat * torch.log(torch.clamp(similarity, min=1e-10, max=1.0)) -
    #                                 (1 - label_mat) * torch.log(torch.clamp(1 - similarity, min=1e-10, max=1.0)))
    #     # result = cross_entropy.sum()/max(torch.sum(mask_mat), 1)
    #     result = cross_entropy.sum() / torch.max(torch.sum(mask_mat).float(), torch.tensor(1.0, device=mask_mat.device))
    #     return result

    # def forward(self, label_vec, mask_vec, y_pred):  # gamma: weight of clustering loss
    #     # 创建标签矩阵
    #     label_vec = label_vec(mask_vec==1)
    #     label_mat = label_vec.view(-1, 1) - label_vec.view(1, -1)
    #     label_mat = (label_mat == 0).float()
    #
    #     # 归一化
    #     y_pred = y_pred(mask_vec==1)
    #     normalize_discriminate = F.normalize(y_pred, p=2, dim=1)
    #
    #     # 计算相似度
    #     similarity = torch.matmul(normalize_discriminate, normalize_discriminate.t())
    #
    #     # 计算交叉熵
    #     cross_entropy = (-label_mat * torch.log(torch.clamp(similarity, min=1e-10, max=1.0)) -
    #                                 (1 - label_mat) * torch.log(torch.clamp(1 - similarity, min=1e-10, max=1.0)))
    #     result = cross_entropy.mean()
    #     return result

    def forward(self, label_vec, mask_vec, y_pred):
        # 仅保留 mask_vec 为 0 的部分
        selected_labels = label_vec[mask_vec == 0]
        selected_preds = y_pred[mask_vec == 0]

        # 创建标签矩阵：1 表示同类，0 表示不同类
        label_mat = (selected_labels.view(-1, 1) == selected_labels.view(1, -1)).float()

        # 归一化 y_pred
        normalized_preds = F.normalize(selected_preds, p=2, dim=1)

        # 计算相似度矩阵
        similarity = torch.matmul(normalized_preds, normalized_preds.t())

        # 计算掩码和交叉熵
        # 对数操作添加小的数值以避免 NaN
        cross_entropy = (-label_mat * torch.log(similarity.clamp(min=1e-10)) -
                         (1 - label_mat) * torch.log((1 - similarity).clamp(min=1e-10)))

        # 计算结果平均值
        result = cross_entropy.mean()

        return result

def cul_batch_kl(csv_batch,cu, n_clusters, U):
    csv_batch = np.array(csv_batch)
    n_batch = len(np.unique(csv_batch))
    bt = torch.from_numpy(csv_batch).to(torch.int64)
    bt_count = torch.bincount(bt)
    qt = bt_count/bt.numel()
    B = torch.Tensor(n_clusters,bt.numel()) #4*5
    B.copy_(bt)
    B = B.t()
    Z = torch.zeros(bt.numel(),n_clusters)
    su = U.sum(axis = 0)
    tensor_list = list()
    for i in range(0, n_batch):
        u0 = torch.where(B==i,cu,Z)
        su0 = u0.sum(axis = 0)
        pb0 = torch.div(su0,su)
        tensor_list.append(pb0)
    pb = torch.stack(tensor_list)
    Q = torch.Tensor(n_clusters,qt.numel())
    Q = Q.copy_(qt).t()
    kl = pb * torch.log(torch.div(pb+1e-6,Q+1e-6))
    kl_sum = kl.sum().item()
    return kl_sum