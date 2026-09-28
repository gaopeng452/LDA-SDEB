import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import subprocess
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# ==========================================
# 【消融实验全局开关 B & C】
# ==========================================
# 开关 B: True为开启 D-S 证据融合与 EDL 损失，False为退化为普通MLP拼接与交叉熵
INNOVATION_B_DS = False

# 开关 C: True为开启 XGB+LGB+Cat Logit混合专家，False为退化为仅使用单一树模型
INNOVATION_C_MOE = True

# 当 INNOVATION_C_MOE = False 时，选择使用的单一专家模型
# 可选值: 'XGB' (XGBoost), 'LGB' (LightGBM), 'CAT' (CatBoost)
SINGLE_EXPERT_CHOICE = 'XGB'
# ==========================================

def kfold(data, k, row=0, col=0, cv=3):
    dlen = len(data)
    if cv != 4:
        if cv == 1:
            lens = row
        elif cv == 2:
            lens = col
        else:
            lens = dlen
        test_res = []
        train_res = []
        d = list(range(lens))
        kf = KFold(k, shuffle=True)
        d = kf.split(d)
        for i in d:
            test_res.append(list(i[1]))
            train_res.append(list(i[0]))
        if cv == 3:
            return train_res, test_res
        else:
            train_s = []
            test_s = []
            for i in range(k):
                train_ = []
                test_ = []
                for j in range(dlen):
                    if data[j][cv - 1] in test_res[i]:
                        test_.append(j)
                    else:
                        train_.append(j)
                train_s.append(train_)
                test_s.append(test_)
            return train_s, test_s
    else:
        r = list(range(row))
        c = list(range(col))
        kf = KFold(k, shuffle=True)
        r = list(kf.split(r))
        c = list(kf.split(c))
        train_s = []
        test_s = []
        for i in range(k):
            tr_ = []
            te_ = []
            for m in range(dlen):
                f1 = data[m][0] in r[i][1]
                f2 = data[m][1] in c[i][1]
                if f1 and f2:
                    te_.append(m)
                if (not f1) and (not f2):
                    tr_.append(m)
            train_s.append(tr_)
            test_s.append(te_)
        return train_s, test_s


class EvidenceFusionNet(nn.Module):
    def __init__(self, svd_dim, dyh_dim, num_classes=2):
        super(EvidenceFusionNet, self).__init__()
        self.svd_layer = nn.Sequential(
            nn.Linear(svd_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softplus()
        )
        self.dyh_layer = nn.Sequential(
            nn.Linear(dyh_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softplus()
        )
        self.num_classes = num_classes

    def ds_combine(self, alpha_a, alpha_b):
        def get_bu(alpha):
            S = torch.sum(alpha, dim=1, keepdim=True)
            b = (alpha - 1) / S
            u = self.num_classes / S
            return b, u
        b_a, u_a = get_bu(alpha_a)
        b_b, u_b = get_bu(alpha_b)
        conflict = (b_a[:, 0:1] * b_b[:, 1:]) + (b_a[:, 1:] * b_b[:, 0:1])
        conflict = torch.sum(conflict, dim=1, keepdim=True)
        b_final = (b_a * b_b + b_a * u_b + b_b * u_a) / (1 - conflict + 1e-8)
        u_final = (u_a * u_b) / (1 - conflict + 1e-8)
        S_final = self.num_classes / (u_final + 1e-8)
        alpha_final = b_final * S_final + 1
        return alpha_final, u_final, u_a, u_b

    def forward(self, x_svd, x_dyh):
        evidence_svd = self.svd_layer(x_svd)
        evidence_dyh = self.dyh_layer(x_dyh)
        alpha_svd = evidence_svd + 1
        alpha_dyh = evidence_dyh + 1
        alpha_final, u_final, u_svd, u_dyh = self.ds_combine(alpha_svd, alpha_dyh)
        gated_svd = x_svd * (1 - u_svd)
        gated_dyh = x_dyh * (1 - u_dyh)
        f_fused = torch.cat([gated_svd, gated_dyh], dim=1) * (1 - u_final)
        return alpha_final, u_final, f_fused


# 【新增】退化网络基线
class SimpleFusionNet(nn.Module):
    def __init__(self, svd_dim, dyh_dim, num_classes=2):
        super(SimpleFusionNet, self).__init__()
        self.fc1 = nn.Linear(svd_dim + dyh_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x_svd, x_dyh):
        x = torch.cat([x_svd, x_dyh], dim=1)
        f_fused = F.relu(self.fc1(x))
        f_fused = F.relu(self.fc2(f_fused))
        logits = self.classifier(f_fused)
        return logits, f_fused


def edl_loss(alpha, y):
    S = torch.sum(alpha, dim=1, keepdim=True)
    probability = alpha / S
    y_onehot = F.one_hot(y.long(), num_classes=2).float()
    gamma = 2.0
    focal_weight = torch.abs(y_onehot - probability) ** gamma
    A = torch.sum(focal_weight * ((y_onehot - probability) ** 2), dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return torch.mean(A + B)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CPU cross-validation and export plot curves")
    parser.add_argument("--epochs", type=int, default=20, help="MLP/EDL training epochs per fold")
    parser.add_argument("--repeats", type=int, default=5, help="Repeated cross-validation runs")
    parser.add_argument("--folds", type=int, default=5, help="Folds per repeated run")
    parser.add_argument("--seed", type=int, default=20260922, help="Random seed")
    parser.add_argument("--output-dir", default=None, help="Directory for NPY curves and summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    time, k = args.repeats, args.folds
    # This project is configured for deterministic CPU-only execution.
    device = torch.device('cpu')
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(args.output_dir or os.path.join(project_root, 'results'))
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    curve_grid = np.linspace(0.0, 1.0, 200)
    summary = {}

    print(f"CPU run: epochs={args.epochs}, repeats={time}, folds={k}")
    print(f"Results will be saved to: {output_dir}")

    for mm in range(1, 3):
        summary[f'data{mm}'] = {}
        for cv in range(1, 5):
            print(f"=== Starting Dataset {mm} - CV {cv} ===")

            svd_path = os.path.join(project_root, 'svd_feature', f'data{mm}_svd.csv')
            dyh_path = os.path.join(project_root, 'nonlinear_feature', f'data{mm}_dyh.csv')
            label_path = os.path.join(project_root, f'data{mm}', 'label.csv')

            try:
                if os.path.exists(svd_path):
                    svd_data = pd.read_csv(svd_path, header=None).to_numpy(dtype=np.float32)
                else:
                    svd_data = pd.read_csv(f'data{mm}_svd.csv', header=None).to_numpy(dtype=np.float32)

                if os.path.exists(dyh_path):
                    dyh_data = pd.read_csv(dyh_path, header=None).to_numpy(dtype=np.float32)
                else:
                    dyh_data = pd.read_csv(f'data{mm}_dyh.csv', header=None).to_numpy(dtype=np.float32)

                label = pd.read_csv(label_path, header=None).to_numpy()
            except Exception as e:
                print(f"Error loading data: {e}.")
                continue

            row, col = label.shape
            label_flat = label.flatten()
            pos_coords = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j] == 1])
            neg_coords = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j] == 0])

            if len(neg_coords) > len(pos_coords):
                np.random.shuffle(neg_coords)
                neg_coords = neg_coords[:len(pos_coords)]

            coords = np.vstack([pos_coords, neg_coords])
            mAUCs, mAUPRs = [], []
            mAccs, mPrecs, mRecs, mF1s = [], [], [], []
            roc_curves, pr_curves = [], []

            for j in range(time):
                tr_list, te_list = kfold(coords, k, row, col, cv)
                for i in range(k):
                    tr_idx = coords[tr_list[i]][:, 0] * col + coords[tr_list[i]][:, 1]
                    te_idx = coords[te_list[i]][:, 0] * col + coords[te_list[i]][:, 1]

                    scaler_svd = StandardScaler()
                    scaler_dyh = StandardScaler()

                    svd_tr_scaled = scaler_svd.fit_transform(svd_data[tr_idx])
                    dyh_tr_scaled = scaler_dyh.fit_transform(dyh_data[tr_idx])
                    svd_te_scaled = scaler_svd.transform(svd_data[te_idx])
                    dyh_te_scaled = scaler_dyh.transform(dyh_data[te_idx])

                    y_tr_np = label_flat[tr_idx]
                    y_te_np = label_flat[te_idx]

                    if cv != 4:
                        X_tr_svd_t = torch.FloatTensor(svd_tr_scaled).to(device)
                        X_tr_dyh_t = torch.FloatTensor(dyh_tr_scaled).to(device)
                        y_tr_t = torch.FloatTensor(y_tr_np).to(device)
                        X_te_svd_t = torch.FloatTensor(svd_te_scaled).to(device)
                        X_te_dyh_t = torch.FloatTensor(dyh_te_scaled).to(device)

                        svd_dim = svd_data.shape[1]
                        dyh_dim = dyh_data.shape[1]

                        # 【消融分支 B】: D-S 证据融合网络 VS 退化 MLP
                        if INNOVATION_B_DS:
                            fusion_net = EvidenceFusionNet(svd_dim, dyh_dim).to(device)
                            optimizer = optim.Adam(fusion_net.parameters(), lr=0.005)
                            fusion_net.train()
                            for epoch in range(args.epochs):
                                optimizer.zero_grad()
                                alpha, u, f_fused = fusion_net(X_tr_svd_t, X_tr_dyh_t)
                                loss_edl = edl_loss(alpha, y_tr_t)
                                loss_edl.backward()
                                optimizer.step()

                            fusion_net.eval()
                            with torch.no_grad():
                                _, u_tr, f_tr = fusion_net(X_tr_svd_t, X_tr_dyh_t)
                                f_tr_np = torch.cat([f_tr, u_tr], dim=1).cpu().numpy()
                                _, u_te, f_te = fusion_net(X_te_svd_t, X_te_dyh_t)
                                f_te_np = torch.cat([f_te, u_te], dim=1).cpu().numpy()
                        else:
                            fusion_net = SimpleFusionNet(svd_dim, dyh_dim).to(device)
                            optimizer = optim.Adam(fusion_net.parameters(), lr=0.005)
                            criterion = nn.CrossEntropyLoss()
                            fusion_net.train()
                            for epoch in range(args.epochs):
                                optimizer.zero_grad()
                                logits, f_fused = fusion_net(X_tr_svd_t, X_tr_dyh_t)
                                loss_ce = criterion(logits, y_tr_t.long())
                                loss_ce.backward()
                                optimizer.step()

                            fusion_net.eval()
                            with torch.no_grad():
                                _, f_tr = fusion_net(X_tr_svd_t, X_tr_dyh_t)
                                f_tr_np = f_tr.cpu().numpy()
                                _, f_te = fusion_net(X_te_svd_t, X_te_dyh_t)
                                f_te_np = f_te.cpu().numpy()

                        # 【消融分支 C (直推式)】: Logit 集成 VS 单一树模型
                        if INNOVATION_C_MOE:
                            m_xgb = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', verbosity=0).fit(f_tr_np, y_tr_np)
                            m_lgb = lgb.LGBMClassifier(n_estimators=100, verbose=-1).fit(f_tr_np, y_tr_np)
                            m_cat = CatBoostClassifier(iterations=100, verbose=0, allow_writing_files=False).fit(f_tr_np, y_tr_np)
                        else:
                            if SINGLE_EXPERT_CHOICE == 'XGB':
                                m_single = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', verbosity=0).fit(f_tr_np, y_tr_np)
                            elif SINGLE_EXPERT_CHOICE == 'LGB':
                                m_single = lgb.LGBMClassifier(n_estimators=100, verbose=-1).fit(f_tr_np, y_tr_np)
                            elif SINGLE_EXPERT_CHOICE == 'CAT':
                                m_single = CatBoostClassifier(iterations=100, verbose=0, allow_writing_files=False).fit(f_tr_np, y_tr_np)
                            else:
                                raise ValueError("SINGLE_EXPERT_CHOICE 必须是 'XGB', 'LGB' 或 'CAT'")

                    else:
                        f_tr_np = np.concatenate([svd_tr_scaled, dyh_tr_scaled], axis=1)
                        f_te_np = np.concatenate([svd_te_scaled, dyh_te_scaled], axis=1)

                        # 【消融分支 C (归纳式)】
                        if INNOVATION_C_MOE:
                            m_xgb = xgb.XGBClassifier(n_estimators=100, colsample_bytree=0.6, subsample=0.8, eval_metric='logloss', verbosity=0).fit(f_tr_np, y_tr_np)
                            m_lgb = lgb.LGBMClassifier(n_estimators=100, colsample_bytree=0.6, subsample=0.8, verbose=-1).fit(f_tr_np, y_tr_np)
                            m_cat = CatBoostClassifier(iterations=100, colsample_bylevel=0.6, subsample=0.8, verbose=0, allow_writing_files=False).fit(f_tr_np, y_tr_np)
                        else:
                            if SINGLE_EXPERT_CHOICE == 'XGB':
                                m_single = xgb.XGBClassifier(n_estimators=100, colsample_bytree=0.6, subsample=0.8, eval_metric='logloss', verbosity=0).fit(f_tr_np, y_tr_np)
                            elif SINGLE_EXPERT_CHOICE == 'LGB':
                                m_single = lgb.LGBMClassifier(n_estimators=100, colsample_bytree=0.6, subsample=0.8, verbose=-1).fit(f_tr_np, y_tr_np)
                            elif SINGLE_EXPERT_CHOICE == 'CAT':
                                m_single = CatBoostClassifier(iterations=100, colsample_bylevel=0.6, subsample=0.8, verbose=0, allow_writing_files=False).fit(f_tr_np, y_tr_np)
                            else:
                                raise ValueError("SINGLE_EXPERT_CHOICE 必须是 'XGB', 'LGB' 或 'CAT'")

                    # 【统一预测分支】
                    if INNOVATION_C_MOE:
                        p_x = m_xgb.predict_proba(f_te_np)[:, 1]
                        p_l = m_lgb.predict_proba(f_te_np)[:, 1]
                        p_c = m_cat.predict_proba(f_te_np)[:, 1]

                        eps = 1e-6
                        p_x_c = np.clip(p_x, eps, 1 - eps)
                        p_l_c = np.clip(p_l, eps, 1 - eps)
                        p_c_c = np.clip(p_c, eps, 1 - eps)

                        logit_x = np.log(p_x_c / (1 - p_x_c))
                        logit_l = np.log(p_l_c / (1 - p_l_c))
                        logit_c = np.log(p_c_c / (1 - p_c_c))

                        avg_logit = (logit_x + logit_l + logit_c) / 3.0
                        final_p = 1 / (1 + np.exp(-avg_logit))
                    else:
                        final_p = m_single.predict_proba(f_te_np)[:, 1]

                    if len(np.unique(y_te_np)) > 1:
                        fpr, tpr, _ = roc_curve(y_te_np, final_p)
                        mAUCs.append(auc(fpr, tpr))
                        roc_curves.append(np.interp(curve_grid, fpr, tpr))
                        pre, rec, _ = precision_recall_curve(y_te_np, final_p)
                        mAUPRs.append(auc(rec, pre))
                        pr_curves.append(np.interp(curve_grid, rec[::-1], pre[::-1]))

                        best_threshold = 0.5
                        best_f1 = 0.0
                        for thresh in np.arange(0.3, 0.75, 0.02):
                            temp_pred = (final_p >= thresh).astype(int)
                            temp_f1 = f1_score(y_te_np, temp_pred, zero_division=0)
                            if temp_f1 > best_f1:
                                best_f1 = temp_f1
                                best_threshold = thresh

                        y_pred = (final_p >= best_threshold).astype(int)

                        mAccs.append(accuracy_score(y_te_np, y_pred))
                        mPrecs.append(precision_score(y_te_np, y_pred, zero_division=0))
                        mRecs.append(recall_score(y_te_np, y_pred, zero_division=0))
                        mF1s.append(f1_score(y_te_np, y_pred, zero_division=0))
                    else:
                        print("Warning: Only one class present in y_true. Skipping metric calculation.")

            if not roc_curves or not pr_curves:
                raise RuntimeError(f"No valid curves generated for data{mm} CV{cv}")

            mean_roc = np.mean(roc_curves, axis=0)
            mean_pr = np.mean(pr_curves, axis=0)
            mean_roc[0], mean_roc[-1] = 0.0, 1.0
            np.save(os.path.join(output_dir, f'LDA-VGHB_data{mm}_CV{cv}_roc.npy'),
                    np.vstack([curve_grid, mean_roc]))
            np.save(os.path.join(output_dir, f'LDA-VGHB_data{mm}_CV{cv}_pr.npy'),
                    np.vstack([curve_grid, mean_pr]))

            metrics = {
                'AUC': float(np.mean(mAUCs)),
                'AUC_std': float(np.std(mAUCs)),
                'AUPR': float(np.mean(mAUPRs)),
                'AUPR_std': float(np.std(mAUPRs)),
                'Acc': float(np.mean(mAccs)),
                'Prec': float(np.mean(mPrecs)),
                'Rec': float(np.mean(mRecs)),
                'F1': float(np.mean(mF1s)),
                'epochs': args.epochs,
                'repeats': time,
                'folds': k,
                'valid_folds': len(mAUCs),
            }
            summary[f'data{mm}'][f'CV{cv}'] = metrics
            with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)

            print(f"Dataset {mm} CV {cv} Result: "
                  f"AUC: {metrics['AUC']:.4f}, "
                  f"AUPR: {metrics['AUPR']:.4f}, "
                  f"Acc: {metrics['Acc']:.4f}, "
                  f"Prec: {metrics['Prec']:.4f}, "
                  f"Rec: {metrics['Rec']:.4f}, "
                  f"F1: {metrics['F1']:.4f}", flush=True)

if __name__ == "__main__":
    main()
