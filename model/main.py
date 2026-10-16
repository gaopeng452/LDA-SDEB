from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from data_utils import (
    DatasetBundle,
    cartesian_pair_features,
    integrated_similarities,
    load_dataset,
    node_profiles,
    similarity_target,
    truncated_svd_node_features,
)
from evaluation import compute_metrics, summarize_metrics, train_logit_blending_ensemble
from model_utils import train_dynamic_hypergraph, train_evidence_fusion
from splits import generate_paper_splits


DATASET_IDS = (1, 2)
CV_MODES = (1, 2, 3, 4)
# Paper protocol: repeat each 5-fold cross-validation setting 20 times
# (100 fold-level evaluations per dataset/CV mode).
REPEATS = 20
FOLDS = 5
SEED = 20261115
SVD_COMPONENTS = 32
DDH_EPOCHS = 100
FUSION_EPOCHS = 50
LEARNING_RATE = 0.005
TREE_ESTIMATORS = 500
THREADS = 1
RESULT_PATH = Path(__file__).resolve().parent / "results" / "results.json"


def pair_rows(pairs: np.ndarray, disease_count: int) -> np.ndarray:
    return pairs[:, 0] * disease_count + pairs[:, 1]


def build_paper_views(
    dataset_id: int,
    seed: int,
    device: torch.device,
) -> tuple[DatasetBundle, np.ndarray, np.ndarray]:
    bundle = load_dataset(dataset_id)
    lnc_svd, disease_svd = truncated_svd_node_features(
        bundle.associations,
        components=SVD_COMPONENTS,
    )
    lnc_profile, disease_profile = node_profiles(bundle.associations)
    lnc_embedding = train_dynamic_hypergraph(
        lnc_profile,
        similarity_target(lnc_profile),
        seed=seed + dataset_id * 1009,
        epochs=DDH_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
    )
    disease_embedding = train_dynamic_hypergraph(
        disease_profile,
        similarity_target(disease_profile),
        seed=seed + dataset_id * 1009 + 1,
        epochs=DDH_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
    )
    lnc_similarity, disease_similarity = integrated_similarities(bundle)
    lnc_ddh = np.concatenate((lnc_embedding, lnc_similarity), axis=1)
    disease_ddh = np.concatenate((disease_embedding, disease_similarity), axis=1)
    svd_view = cartesian_pair_features(lnc_svd, disease_svd)
    ddh_view = cartesian_pair_features(lnc_ddh, disease_ddh)
    return bundle, svd_view, ddh_view


def run() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {
        "created_at": datetime.now().astimezone().isoformat(),
        "parameters": {
            "repeats": REPEATS,
            "folds": FOLDS,
            "seed": SEED,
            "svd_components": SVD_COMPONENTS,
            "ddh_channels": 4,
            "gumbel_temperature": 1.0,
            "ddh_epochs": DDH_EPOCHS,
            "fusion_epochs": FUSION_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "estimators": TREE_ESTIMATORS,
            "threshold": 0.5,
            "device": str(device),
        },
        "experiments": [],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for dataset_id in DATASET_IDS:
        print(f"\n=== building data{dataset_id} LDA-SDEB views ===", flush=True)
        bundle, svd_view, ddh_view = build_paper_views(dataset_id, SEED, device)
        disease_count = bundle.associations.shape[1]
        print(f"SVD={svd_view.shape}, DDH={ddh_view.shape}, device={device}", flush=True)

        for cv_mode in CV_MODES:
            print(f"\n=== data{dataset_id} CV{cv_mode} ===", flush=True)
            splits = generate_paper_splits(
                bundle.associations,
                dataset_id=dataset_id,
                cv_mode=cv_mode,
                repeats=REPEATS,
                n_splits=FOLDS,
                seed=SEED,
            )
            rows = []
            for split in splits:
                fold_seed = (
                    SEED
                    + dataset_id * 1_000_003
                    + cv_mode * 10_007
                    + split.repeat * 101
                    + split.fold
                )
                train_index = pair_rows(split.train_pairs, disease_count)
                test_index = pair_rows(split.test_pairs, disease_count)
                svd_scaler = StandardScaler().fit(svd_view[train_index])
                ddh_scaler = StandardScaler().fit(ddh_view[train_index])
                train_svd = svd_scaler.transform(svd_view[train_index]).astype(np.float32)
                test_svd = svd_scaler.transform(svd_view[test_index]).astype(np.float32)
                train_ddh = ddh_scaler.transform(ddh_view[train_index]).astype(np.float32)
                test_ddh = ddh_scaler.transform(ddh_view[test_index]).astype(np.float32)

                train_fused, test_fused = train_evidence_fusion(
                    train_svd,
                    train_ddh,
                    split.train_pairs[:, 2],
                    test_svd,
                    test_ddh,
                    seed=fold_seed,
                    epochs=FUSION_EPOCHS,
                    learning_rate=LEARNING_RATE,
                    device=device,
                )
                probability = train_logit_blending_ensemble(
                    train_fused,
                    split.train_pairs[:, 2],
                    test_fused,
                    seed=fold_seed + 1,
                    estimators=TREE_ESTIMATORS,
                    threads=THREADS,
                )
                metric = compute_metrics(split.test_pairs[:, 2], probability, threshold=0.5)
                metric.update(
                    {
                        "repeat": split.repeat,
                        "fold": split.fold,
                        "train_size": int(len(split.train_pairs)),
                        "test_size": int(len(split.test_pairs)),
                    }
                )
                rows.append(metric)
                print(
                    f"repeat={split.repeat + 1:02d} fold={split.fold + 1} "
                    f"AUC={metric['auc']:.4f} AUPR={metric['aupr']:.4f}",
                    flush=True,
                )

            summary = summarize_metrics(rows)
            output["experiments"].append(
                {
                    "dataset_id": dataset_id,
                    "cv_mode": cv_mode,
                    "feature_name": "svd_ddh_ds_fused",
                    "feature_dimensions": {
                        "svd": int(svd_view.shape[1]),
                        "ddh": int(ddh_view.shape[1]),
                        "fused": int(svd_view.shape[1] + ddh_view.shape[1]),
                    },
                    "fold_results": rows,
                    "summary": summary,
                }
            )
            RESULT_PATH.write_text(
                json.dumps(output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"SUMMARY data{dataset_id} CV{cv_mode}: "
                f"AUC={summary['auc']['mean']:.4f} AUPR={summary['aupr']['mean']:.4f}",
                flush=True,
            )
    return output


if __name__ == "__main__":
    run()
