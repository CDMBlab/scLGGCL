import anndata
import numpy as np
import scanpy as  sc
import torch
import scanpy
import h5py
import random


def normalize(adata, highly_genes = None, size_factors=True, normalize_input=True, logtrans_input=True):
    #过滤低质量细胞样本：过滤在少于10个细胞中表达，或一个细胞中表达少于1个基因的细胞样本
    sc.pp.filter_genes(adata, min_counts=1)#过滤基因，在少于10个细胞样本上表达的基因
    sc.pp.filter_cells(adata, min_counts=1)#过滤细胞，少于1个表达基因的细胞
    if size_factors or normalize_input or logtrans_input:#size_factors文库大小：每一个细胞所有基因表达量的总和。
        #判断输入数据是否经过文库大小标准化或标准化或log转换
        adata.raw = adata    #存储数据，如果不想adata.X里面的值修改，就使用copy()
    else:
        adata.raw = adata   #保存原始数据

    if size_factors:
        sc.pp.normalize_per_cell(adata) #通过所有基因的总数对每个细胞进行归一化，以便每个细胞都有规范化后的总数相同。
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)  #每个细胞的大小因子
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)  # 对数化

    if highly_genes != None:
        # # 去除重复的基因名称
        # adata.var_names = pd.unique(adata.var_names)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes)

        # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)  #识别特异基因，确定高变基因；在细胞与细胞间进行比较，选择表达量差别最大的基因。data：AnnData Matrix，行对应细胞，列对应基因
        #n_top_genes：要保留的高变基因的数量

    if normalize_input:
        sc.pp.scale(adata)  #将每个基因缩放到单位方差

    adata.raw.var['highly_variable'] = adata.var['highly_variable']
    adata = adata[:, adata.var['highly_variable']]
    dataMat = adata.X
    rawData = adata.raw[:, adata.raw.var['highly_variable']].X

    return adata, dataMat, rawData

def read_real1(filename):
    print('loading data!')
    adata = anndata.read(filename)
    mat= adata.X
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    obs = adata.obs
    cell_name = np.array(obs["cell_ontology_class"])
    if (cell_name == "").sum() > 0:
        cell_name[cell_name == ""] = "unknown_class"
    return X, cell_name


def read_real2(filename):
    adata = anndata.read(filename)
    mat= adata.X
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    obs = adata.obs
    cell_name = np.array(obs["cell_ontology_class"])
    if (cell_name == "").sum() > 0:
        cell_name[cell_name == ""] = "unknown_class"

    var_name = adata.var_names
    genes_name = np.array(var_name)
    return X, cell_name, genes_name


def preprocess(X, Y, batch_label, highly_genes=2000, dropout=0):
    adata = sc.AnnData(X)  # scanpy中常见的数据结构是AnnData，它是一个用于存储数据的对象，
    adata.obs["celltype"] = Y  # adata.obs	观测量	pandas Dataframe
    adata.obs["batch"] = batch_label  # 批次标签
    adata, dataMat, rawData = normalize(adata, highly_genes=highly_genes, size_factors=True, normalize_input=False,
                      logtrans_input=True)  # 标准化

    return adata, rawData, dataMat


def dist_2_label(p):
    _, label = torch.max(p, dim=1)
    return label.data.cpu().numpy()

#constructing the cell-cell graph
def adata_knn(adata, method, knn, n_neighbors, metric='cosine'):
    # if adata.shape[0] >=10000:
    #     scanpy.pp.pca(adata, n_comps=50)
    #     n_pcs = 50
    # else:
    #     n_pcs=32
    n_pcs = 32
    if method == 'umap':
        scanpy.pp.neighbors(adata, method = method, metric=metric,
                            knn=knn, n_pcs=n_pcs, n_neighbors=n_neighbors)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']
    elif method == 'gauss':
        scanpy.pp.neighbors(adata, method = 'gauss', metric=metric,
                            knn=knn, n_pcs=n_pcs, n_neighbors=n_neighbors)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']
    return adj, r_adj

def read_simu(dataname):
    data_path = "./Data/simulation/" + dataname
    data_mat = h5py.File(data_path)
    x = np.array(data_mat["X"])
    y = np.array(data_mat["Y"])
    batch = np.array(data_mat["B"])
    return x, y, batch

def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    # result = result.max(torch.zeros(result.shape).cuda())
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    # result = torch.max(result, result.t())
    return result

def adaptive_loss(D, sigma=1):
    return (1 + sigma) * D * D / (D + sigma)

def update_U(Z, n_clusters, sigma=1, gamma =1):
    idx = random.sample(list(range(Z.shape[1])), n_clusters)
    centroids = Z[:, idx] + 10 ** -8
    distances = adaptive_loss(distance(Z, centroids, False), sigma)
    U = torch.exp(-distances / gamma)
    U = U / U.sum(dim=1).reshape([-1, 1])
    return U