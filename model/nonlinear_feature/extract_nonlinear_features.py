import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 【消融实验全局开关 A】
# True: 开启原版动态解耦超图 (Gumbel硬采样 + 正交解耦)
# False: 退化为静态图软注意力 (无离散结构 + 无解耦)
# ==========================================
INNOVATION_A_DYH = False


class DisentangledDynamicHypergraph(nn.Module):
    def __init__(self, in_features, out_features=64, K=4, tau=1.0):
        super(DisentangledDynamicHypergraph, self).__init__()
        self.K = K
        self.tau = tau
        self.out_channels = out_features // K

        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_features, self.out_channels)) for _ in range(K)
        ])

        for w in self.weights:
            nn.init.orthogonal_(w)

        self.spatial_proj = nn.Linear(in_features, in_features, bias=False)
        nn.init.orthogonal_(self.spatial_proj.weight)

    def forward(self, Z):
        Z_spatial = torch.tanh(self.spatial_proj(Z))
        Sim = torch.matmul(Z_spatial, Z_spatial.t())

        I = torch.eye(Z.size(0)).to(Z.device)
        Sim = Sim + I * 2.0

        # 【消融点 A】: 图结构生成方式
        if INNOVATION_A_DYH:
            H_learn = F.gumbel_softmax(Sim, tau=self.tau, hard=True, dim=-1)
        else:
            H_learn = F.softmax(Sim, dim=-1)

        D_v = torch.sum(H_learn, dim=1)
        D_v_inv = 1.0 / (D_v + 1e-8)
        D_v_inv_mat = torch.diag(D_v_inv)

        Z_k_list = []
        for k in range(self.K):
            ZW = torch.matmul(Z, self.weights[k])
            H_ZW = torch.matmul(H_learn, ZW)
            Z_k = torch.matmul(D_v_inv_mat, H_ZW)
            Z_k = F.relu(Z_k)
            Z_k_list.append(Z_k)

        # 【消融点 A】: 正交解耦约束
        ortho_loss = 0.0
        if INNOVATION_A_DYH:
            for i in range(self.K):
                for j in range(i + 1, self.K):
                    z_i = F.normalize(Z_k_list[i], p=2, dim=1)
                    z_j = F.normalize(Z_k_list[j], p=2, dim=1)
                    ortho_loss += torch.mean(torch.abs(torch.sum(z_i * z_j, dim=1)))
        else:
            ortho_loss = torch.tensor(0.0).to(Z.device)

        X_nonlinear = torch.cat(Z_k_list, dim=1)
        return X_nonlinear, ortho_loss


def train_and_extract(features, adj_target, entity_name, epochs=100, lr=0.005):
    # Keep feature extraction on CPU even when a CUDA-capable GPU is present.
    device = torch.device('cpu')
    print(f"[{entity_name}] Training Dynamic Hypergraph Model on {device}...")

    features_t = torch.FloatTensor(features).to(device)
    adj_t = torch.FloatTensor(adj_target).to(device)

    model = DisentangledDynamicHypergraph(in_features=features.shape[1], out_features=64, K=4, tau=1.0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out_feat, ortho_loss = model(features_t)

        pred_adj = torch.matmul(out_feat, out_feat.t())
        recon_loss = F.mse_loss(pred_adj, adj_t)

        loss = recon_loss + 0.1 * ortho_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"  -> Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, Ortho: {ortho_loss.item():.4f})")

    model.eval()
    with torch.no_grad():
        final_feat, _ = model(features_t)

    return final_feat.cpu().numpy()


def extract_nonlinear_for_dataset(mm):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    data_path = os.path.join(project_root, f'data{mm}', 'label.csv')
    lnc_sim_path = os.path.join(project_root, f'data{mm}', 'lncRNA_sim.csv')
    dis_sim_path = os.path.join(project_root, f'data{mm}', 'disease_sim.csv')

    if not os.path.exists(data_path):
        print(f"[Error] File not found: {data_path}")
        return

    print(f"\nProcessing Dataset {mm} from: {data_path}")

    try:
        A = pd.read_csv(data_path, header=None).values
    except Exception as e:
        print(f"[Error] Failed to read csv: {e}")
        return

    A_norm = (A - A.mean()) / (A.std() + 1e-8)
    adj_lnc = np.nan_to_num(np.corrcoef(A))
    adj_dis = np.nan_to_num(np.corrcoef(A.T))


    lnc_feat = train_and_extract(A_norm, adj_lnc, "LncRNA", epochs=100)
    dis_feat = train_and_extract(A_norm.T, adj_dis, "Disease", epochs=100)

    if os.path.exists(lnc_sim_path) and os.path.exists(dis_sim_path):
        print(f"  -> [DyH] Found biological similarity matrices, concatenating...")
        lnc_sim = pd.read_csv(lnc_sim_path, header=None).values
        dis_sim = pd.read_csv(dis_sim_path, header=None).values
        lnc_feat = np.concatenate([lnc_feat, lnc_sim], axis=1)
        dis_feat = np.concatenate([dis_feat, dis_sim], axis=1)
    else:
        print(f"  -> [DyH] Warning: lncRNA_sim.csv or disease_sim.csv not found.")

    row_count, col_count = A.shape
    pair_features = []
    for i in range(row_count):
        for j in range(col_count):
            pair_feat = np.concatenate([lnc_feat[i], dis_feat[j]])
            pair_features.append(pair_feat)

    output_path = os.path.join(current_dir, f'data{mm}_dyh.csv')
    pd.DataFrame(pair_features).to_csv(output_path, index=False, header=False)
    print(f"Dataset {mm} Nonlinear features saved to {output_path}")

if __name__ == "__main__":
    for dataset_idx in range(1, 3):
        extract_nonlinear_for_dataset(dataset_idx)
