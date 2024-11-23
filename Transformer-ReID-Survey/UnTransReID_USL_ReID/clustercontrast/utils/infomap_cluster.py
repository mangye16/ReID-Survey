import numpy as np
from tqdm import tqdm
import infomap
import faiss
import math
import multiprocessing as mp
from clustercontrast.utils.infomap_utils import Timer




def l2norm(vec):
    """
    归一化
    :param vec:
    :return:
    """
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb：每一个顶点对应一个类
    lb2idxs：每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


class knn_faiss():
    """
    内积暴力循环
    归一化特征的内积等价于余弦相似度
    """

    def __init__(self, feats, k, knn_method='faiss-cpu', verbose=True):
        self.verbose = verbose

        with Timer('[{}] build index {}'.format(knn_method, k), verbose):
            feats = feats.astype('float32')
            size, dim = feats.shape
            if knn_method == 'faiss-gpu':
                i = math.ceil(size / 1000000)
                if i > 1:
                    i = (i - 1) * 4
                res = faiss.StandardGpuResources()
                res.setTempMemory(i * 1024 * 1024 * 1024)
                index = faiss.GpuIndexFlatIP(res, dim)
            else:
                index = faiss.IndexFlatIP(dim)
            index.add(feats)

        with Timer('[{}] query topk {}'.format(knn_method, k), verbose):
            sims, nbrs = index.search(feats, k=k)
            self.knns = [(np.array(nbr, dtype=np.int32),
                          1 - np.array(sim, dtype=np.float32))
                         for nbr, sim in zip(nbrs, sims)]

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# 构造边
def get_links(single, links, nbrs, dists, min_sim):
    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        for j in range(0, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - min_sim:
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
            else:
                break
        # 统计孤立点
        if count == 0:
            single.append(i)
    return single, links


def cluster_by_infomap(nbrs, dists, min_sim, cluster_num=2):
    """
    基于infomap的聚类
    :param nbrs:
    :param dists:
    :param pred_label_path:
    :return:
    """
    single = []
    links = {}
    with Timer('get links', verbose=True):
        single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists, min_sim=min_sim)

    infomapWrapper = infomap.Infomap("--two-level --directed")
    for (i, j), sim in tqdm(links.items()):
        _ = infomapWrapper.addLink(int(i), int(j), sim)

    # 聚类运算
    infomapWrapper.run()

    label2idx = {}
    idx2label = {}

    # 聚类结果统计
    for node in infomapWrapper.iterTree():
        # node.physicalId 特征向量的编号
        # node.moduleIndex() 聚类的编号
        if node.moduleIndex() not in label2idx:
            label2idx[node.moduleIndex()] = []
        label2idx[node.moduleIndex()].append(node.physicalId)

    node_count = 0
    for k, v in label2idx.items():
        if k == 0:
            each_index_list = v[2:]
            node_count += len(each_index_list)
            label2idx[k] = each_index_list
        else:
            each_index_list = v[1:]
            node_count += len(each_index_list)
            label2idx[k] = each_index_list

        for each_index in each_index_list:
            idx2label[each_index] = k

    keys_len = len(list(label2idx.keys()))
    # 孤立点放入到结果中
    for single_node in single:
        idx2label[single_node] = keys_len
        label2idx[keys_len] = [single_node]
        keys_len += 1
        node_count += 1

    # 孤立点个数
    print("孤立点数：{}".format(len(single)))

    idx_len = len(list(idx2label.keys()))
    assert idx_len == node_count, 'idx_len not equal node_count!'

    print("总节点数：{}".format(idx_len))

    old_label_container = set()
    for each_label, each_index_list in label2idx.items():
        if len(each_index_list) <= cluster_num:
            for each_index in each_index_list:
                idx2label[each_index] = -1
        else:
            old_label_container.add(each_label)

    old2new = {old_label: new_label for new_label, old_label in enumerate(old_label_container)}

    for each_index, each_label in idx2label.items():
        if each_label == -1:
            continue
        idx2label[each_index] = old2new[each_label]

    pre_labels = intdict2ndarray(idx2label)

    print("总类别数：{}/{}".format(keys_len, len(set(pre_labels)) - (1 if -1 in pre_labels else 0)))

    return pre_labels


def get_dist_nbr(features, k=80, knn_method='faiss-cpu'):
    index = knn_faiss(feats=features, k=k, knn_method=knn_method)
    knns = index.get_knns()
    dists, nbrs = knns2ordered_nbrs(knns)
    return dists, nbrs



