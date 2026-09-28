"""Fast CPU-only end-to-end check using a small balanced data sample."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from main import SimpleFusionNet


def main():
    root = Path(__file__).resolve().parent
    label = pd.read_csv(root / "data1" / "label.csv", header=None).to_numpy().ravel()
    svd = pd.read_csv(root / "svd_feature" / "data1_svd.csv", header=None).to_numpy(dtype=np.float32)
    dyh = pd.read_csv(root / "nonlinear_feature" / "data1_dyh.csv", header=None).to_numpy(dtype=np.float32)

    if not (len(label) == len(svd) == len(dyh)):
        raise RuntimeError("Feature and label row counts do not match")

    positive = np.flatnonzero(label == 1)[:64]
    negative = np.flatnonzero(label == 0)[:64]
    indices = np.concatenate([positive, negative])
    y = label[indices].astype(np.int64)
    x_svd = StandardScaler().fit_transform(svd[indices]).astype(np.float32)
    x_dyh = StandardScaler().fit_transform(dyh[indices]).astype(np.float32)

    device = torch.device("cpu")
    net = SimpleFusionNet(x_svd.shape[1], x_dyh.shape[1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()
    xs = torch.from_numpy(x_svd).to(device)
    xd = torch.from_numpy(x_dyh).to(device)
    yt = torch.from_numpy(y).to(device)

    net.train()
    for _ in range(2):
        optimizer.zero_grad()
        logits, fused = net(xs, xd)
        loss = criterion(logits, yt)
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        _, fused = net(xs, xd)
    features = fused.cpu().numpy()

    models = [
        XGBClassifier(n_estimators=5, eval_metric="logloss", verbosity=0),
        LGBMClassifier(n_estimators=5, verbose=-1),
        CatBoostClassifier(iterations=5, verbose=0, allow_writing_files=False),
    ]
    for model in models:
        probabilities = model.fit(features, y).predict_proba(features)
        if probabilities.shape != (len(y), 2):
            raise RuntimeError(f"Unexpected output from {type(model).__name__}")

    print(f"PASS: CPU pipeline processed {len(y)} real samples; loss={loss.item():.6f}")
    print(f"SVD shape={svd.shape}, nonlinear shape={dyh.shape}, CUDA={torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
