import numpy as np
from scipy import sparse
import torch 
import time 
from tqdm import tqdm 

from evaluate import eval_func, euclidean_dist

def calculate_V(initial_rank, all_feature_len, dis_i_qg, i,  k1):
    # dis_i_qg = euclidean_dist(torch.tensor([all_feature[i].numpy()]), all_feature).numpy()

    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    # print(forward_k_neigh_index)
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]

    fi = np.where(backward_k_neigh_index == i)[0]
    k_reciprocal_index = forward_k_neigh_index[fi]
    k_reciprocal_expansion_index = k_reciprocal_index
    for j in range(len(k_reciprocal_index)):
        candidate = k_reciprocal_index[j]
        candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2.)) + 1]
        candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                           :int(np.around(k1 / 2.)) + 1]
        fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
        candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
        if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                candidate_k_reciprocal_index):
            k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

    k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
    # print(k_reciprocal_expansion_index)
    weight = np.exp(-dis_i_qg[k_reciprocal_expansion_index])
    # print(weight)
    V = np.zeros(( all_feature_len)).astype(np.float32)
    V[k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    return V, k_reciprocal_expansion_index, weight


def re_ranking_batch(all_feature, q_num, k1, k2, lambda_value, len_slice=1000):

    # calculate (q+g)*(q+g)
    initial_rank = np.zeros((len(all_feature), k1+1)).astype(np.int32)

    original_dist = np.zeros((q_num, len(all_feature)))

    s_time = time.time()

    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)

    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            dis_i_qg = euclidean_dist(all_feature[i*len_slice:(i+1)*len_slice], all_feature).data.cpu().numpy()
            initial_i_rank = np.argpartition(dis_i_qg, range(1, k1 + 1), ).astype(np.int32)[:, :k1 + 1]
            initial_rank[i*len_slice:(i+1)*len_slice] = initial_i_rank
            pbar.update(1)
    # print(initial_rank[0])

    end_time = time.time()
    print("rank time : %s" % (end_time-s_time))

    all_V = []

    s_time = time.time()

    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)


    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            dis_i_qg = euclidean_dist(all_feature[i * len_slice:(i + 1) * len_slice], all_feature).data.cpu().numpy()
            for ks in range(dis_i_qg.shape[0]):
                r_k = i*len_slice+ks
                dis_i_qg[ks] = np.power(dis_i_qg[ks], 2).astype(np.float32)
                dis_i_qg[ks] = 1. * dis_i_qg[ks] / np.max(dis_i_qg[ks])
                if r_k < q_num:
                    original_dist[r_k] = dis_i_qg[ks]
                V ,k_reciprocal_expansion_index, weight = calculate_V(initial_rank, len(all_feature), dis_i_qg[ks], r_k, k1)
                # if r_k == 0:
                #     print(k_reciprocal_expansion_index)
                #     print(weight)
                #     print(dis_i_qg[ks])
                all_V.append(sparse.csr_matrix(V))

            pbar.update(1)

    all_V = sparse.vstack(all_V)
    # print(all_V.getrow(0).toarray())
    end_time = time.time()
    print("calculate V time : %s" % (end_time - s_time))
    # print(all_V.todense()[0])

    all_V_qe = []
    s_time = time.time()
    for i in range(len(all_feature)):
        temp_V = np.zeros((k2, len(all_feature)))
        for l, row_index in enumerate(initial_rank[i, :k2]):
            temp_V[l, :] = all_V.getrow(row_index).toarray()[0]


        V_qe = np.mean(temp_V, axis=0)
        all_V_qe.append(sparse.csr_matrix(V_qe))
    all_V_qe = sparse.vstack(all_V_qe)
    # print(all_V_qe.todense()[0])
    del all_V
    end_time = time.time()
    print("calculate V_qe time : %s" % (end_time - s_time))

    invIndex = []
    for i in range(len(all_feature)):
        invIndex.append(np.where(all_V_qe.getcol(i).toarray().transpose()[0] != 0)[0])
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(q_num):
        temp_min = np.zeros(shape=[1, len(all_feature)], dtype=np.float32)

        indNonZero = np.where(all_V_qe.getrow(i).toarray()[0] != 0)[0]

        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        # print(indImages)
        for j in range(len(indNonZero)):
            # print(indNonZero[j])
            c = all_V_qe.getrow(i).getcol(indNonZero[j]).toarray()[0, 0]
            # print(c)
            # print(indImages[j])

            t_min = np.zeros((indImages[j].shape[0]))
            for kk in range(indImages[j].shape[0]):
                temp_d = all_V_qe.getrow(indImages[j][kk]).getcol(indNonZero[j]).toarray()[0, 0]
                t_min[kk] = np.minimum(c, temp_d)
            # print(t_min)

            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + t_min
            # temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
            #                                                                    V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
    # print(jaccard_dist[0])
    # print(original_dist[0])
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del all_V_qe
    del jaccard_dist
    final_dist = final_dist[:q_num, q_num:]
    return final_dist

def re_ranking_batch_gpu(all_feature, q_num, k1, k2, lambda_value, len_slice=1000):

    # calculate (q+g)*(q+g)
    initial_rank = np.zeros((len(all_feature), k1+1)).astype(np.int32)

    original_dist = np.zeros((q_num, len(all_feature)))
    gpu_features = all_feature.cuda()
    s_time = time.time()

    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)

    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            dis_i_qg = euclidean_dist(gpu_features[i*len_slice:(i+1)*len_slice], gpu_features).data.cpu().numpy()
            initial_i_rank = np.argpartition(dis_i_qg, range(1, k1 + 1), ).astype(np.int32)[:, :k1 + 1]
            initial_rank[i*len_slice:(i+1)*len_slice] = initial_i_rank
            pbar.update(1)
    # print(initial_rank[0])

    end_time = time.time()
    print("rank time : %s" % (end_time-s_time))

    all_V = []

    s_time = time.time()

    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)


    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            dis_i_qg = euclidean_dist(gpu_features[i * len_slice:(i + 1) * len_slice], gpu_features).data.cpu().numpy()
            for ks in range(dis_i_qg.shape[0]):
                r_k = i*len_slice+ks
                dis_i_qg[ks] = np.power(dis_i_qg[ks], 2).astype(np.float32)
                dis_i_qg[ks] = 1. * dis_i_qg[ks] / np.max(dis_i_qg[ks])
                if r_k < q_num:
                    original_dist[r_k] = dis_i_qg[ks]
                V ,k_reciprocal_expansion_index, weight = calculate_V(initial_rank, len(all_feature), dis_i_qg[ks], r_k, k1)
                # if r_k == 0:
                #     print(k_reciprocal_expansion_index)
                #     print(weight)
                #     print(dis_i_qg[ks])
                all_V.append(sparse.csr_matrix(V))

            pbar.update(1)

    all_V = sparse.vstack(all_V)
    # print(all_V.getrow(0).toarray())
    end_time = time.time()
    print("calculate V time : %s" % (end_time - s_time))
    # print(all_V.todense()[0])

    all_V_qe = []
    s_time = time.time()
    for i in range(len(all_feature)):
        temp_V = np.zeros((k2, len(all_feature)))
        for l, row_index in enumerate(initial_rank[i, :k2]):
            temp_V[l, :] = all_V.getrow(row_index).toarray()[0]


        V_qe = np.mean(temp_V, axis=0)
        all_V_qe.append(sparse.csr_matrix(V_qe))
    all_V_qe = sparse.vstack(all_V_qe)
    # print(all_V_qe.todense()[0])
    del all_V
    end_time = time.time()
    print("calculate V_qe time : %s" % (end_time - s_time))

    invIndex = []
    for i in range(len(all_feature)):
        invIndex.append(np.where(all_V_qe.getcol(i).toarray().transpose()[0] != 0)[0])
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    with tqdm(total=q_num) as pbar:
        for i in range(q_num):
            temp_min = np.zeros(shape=[1, len(all_feature)], dtype=np.float32)

            indNonZero = np.where(all_V_qe.getrow(i).toarray()[0] != 0)[0]

            indImages = []
            indImages = [invIndex[ind] for ind in indNonZero]
            # print(indImages)
            for j in range(len(indNonZero)):
                # print(indNonZero[j])
                c = all_V_qe.getrow(i).getcol(indNonZero[j]).toarray()[0, 0]
                # print(c)
                # print(indImages[j])

                t_min = np.zeros((indImages[j].shape[0]))
                for kk in range(indImages[j].shape[0]):
                    temp_d = all_V_qe.getrow(indImages[j][kk]).getcol(indNonZero[j]).toarray()[0, 0]
                    t_min[kk] = np.minimum(c, temp_d)
                # print(t_min)

                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + t_min
                # temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                #                                                                    V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
            pbar.update(1)
    # print(jaccard_dist[0])
    # print(original_dist[0])
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del all_V_qe
    del jaccard_dist
    final_dist = final_dist[:q_num, q_num:]
    return final_dist
