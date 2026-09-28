import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds


pca_dim = 32
pca_maxiter = 10_000


def extract_svd_for_dataset(mm):
    # 1. 路径锚定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    data_path = os.path.join(project_root, f'data{mm}', 'label.csv')

    # === 定义相似性矩阵路径 ===
    lnc_sim_path = os.path.join(project_root, f'data{mm}', 'lncRNA_sim.csv')
    dis_sim_path = os.path.join(project_root, f'data{mm}', 'disease_sim.csv')

    if not os.path.exists(data_path):
        print(f"[Error] File not found: {data_path}")
        return

    print(f"Processing SVD for Dataset {mm} from: {data_path}")

    # 读取数据
    try:
        interMatrix = pd.read_csv(data_path, header=None).values
    except Exception as e:
        print(f"[Error] Failed to read csv: {e}")
        return

    interMatrix = interMatrix.astype('float')

    # 执行奇异值分解 (SVD)
    k_val = min(pca_dim, min(interMatrix.shape) - 1)
    U, S_vec, VT = svds(sp.csr_matrix(interMatrix), k=k_val, maxiter=pca_maxiter)

    # 提取特征表示
    lnc_feat = U
    dis_feat = VT.T

    # ==========================================
    # 核心修改：引入生物学相似性特征拼接
    # ==========================================
    if os.path.exists(lnc_sim_path) and os.path.exists(dis_sim_path):
        print(f"  -> [SVD] Found biological similarity matrices, concatenating...")
        lnc_sim = pd.read_csv(lnc_sim_path, header=None).values
        dis_sim = pd.read_csv(dis_sim_path, header=None).values
        # 将拓扑特征与生物学相似性特征在列方向拼接
        lnc_feat = np.concatenate([lnc_feat, lnc_sim], axis=1)
        dis_feat = np.concatenate([dis_feat, dis_sim], axis=1)
    else:
        print(
            f"  -> [SVD] Warning: lncRNA_sim.csv or disease_sim.csv not found. Proceeding with pure topological features.")
    # ==========================================

    # 构造笛卡尔积对特征
    row_count, col_count = interMatrix.shape
    pair_features = []

    for i in range(row_count):
        for j in range(col_count):
            pair_feat = np.concatenate([lnc_feat[i], dis_feat[j]])
            pair_features.append(pair_feat)

    # 保存结果
    output_path = os.path.join(current_dir, f'data{mm}_svd.csv')
    pd.DataFrame(pair_features).to_csv(output_path, index=False, header=False)
    print(f"Dataset {mm} SVD features saved to: {output_path}")


if __name__ == "__main__":
    for dataset_idx in range(1, 3):
        extract_svd_for_dataset(dataset_idx)
