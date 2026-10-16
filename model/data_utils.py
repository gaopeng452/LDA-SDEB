from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    dataset_id: int
    name: str
    associations: np.ndarray
    disease_semantic: np.ndarray
    lnc_functional: np.ndarray
    data_dir: Path


DATASET_PREFIX = {
    1: (
        "lncRNADisease",
        "lncRNADisease-disease semantic similarity matrix.xls",
        "lncRNADisease-lncRNA functional similarity matrix.xls",
    ),
    2: (
        "MNDR",
        "MNDR-disease semantic similarity matrix.xls",
        "MNDR-lncRNA functional similarity matrix.xls",
    ),
}


def default_data_root() -> Path:
    """Return this repository's bundled dataset directory."""
    return Path(__file__).resolve().parent


def load_dataset(dataset_id: int, data_root: Path | None = None) -> DatasetBundle:
    if dataset_id not in DATASET_PREFIX:
        raise ValueError(f"dataset_id must be 1 or 2, got {dataset_id}")
    root = Path(data_root).resolve() if data_root else default_data_root()
    name, disease_file, lnc_file = DATASET_PREFIX[dataset_id]
    data_dir = root / f"data{dataset_id}"
    association_path = data_dir / "label.csv"
    semantic_path = data_dir / disease_file
    functional_path = data_dir / lnc_file
    if not association_path.exists() or not semantic_path.exists() or not functional_path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_id} was not found under {data_dir}. "
            "Place the required data1/data2 directories in the repository root "
            "or pass a data_root directory to load_dataset()."
        )

    associations = pd.read_csv(association_path, header=None).to_numpy(dtype=np.float32)
    # The supplied legacy .xls files have names in their first row and column.
    disease_semantic = (
        pd.read_excel(semantic_path, header=None)
        .iloc[1:, 1:]
        .to_numpy(dtype=np.float32)
    )
    lnc_functional = (
        pd.read_excel(functional_path, header=None)
        .iloc[1:, 1:]
        .to_numpy(dtype=np.float32)
    )
    expected = (associations.shape[1], associations.shape[1])
    if disease_semantic.shape != expected:
        raise ValueError(
            f"Disease semantic matrix has shape {disease_semantic.shape}; expected {expected}."
        )
    lnc_expected = (associations.shape[0], associations.shape[0])
    if lnc_functional.shape != lnc_expected:
        raise ValueError(
            f"lncRNA functional matrix has shape {lnc_functional.shape}; expected {lnc_expected}."
        )
    if not np.isin(associations, (0.0, 1.0)).all():
        raise ValueError("The association matrix must be binary.")

    return DatasetBundle(
        dataset_id=dataset_id,
        name=name,
        associations=associations,
        disease_semantic=disease_semantic,
        lnc_functional=lnc_functional,
        data_dir=data_dir,
    )


def truncated_svd_node_features(
    associations: np.ndarray,
    components: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(associations, dtype=np.float64)
    rank = min(int(components), min(matrix.shape))
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :rank].astype(np.float32), vt.T[:, :rank].astype(np.float32)


def integrated_similarities(bundle: DatasetBundle) -> tuple[np.ndarray, np.ndarray]:
    lnc_gip = gaussian_interaction_profile(bundle.associations)
    disease_gip = gaussian_interaction_profile(bundle.associations.T)
    lnc_integrated = 0.5 * (bundle.lnc_functional + lnc_gip)
    disease_integrated = 0.5 * (bundle.disease_semantic + disease_gip)
    return lnc_integrated.astype(np.float32), disease_integrated.astype(np.float32)


def node_profiles(associations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(associations, dtype=np.float32)
    normalized = (matrix - matrix.mean()) / (matrix.std() + 1e-8)
    return normalized.astype(np.float32), normalized.T.astype(np.float32)


def similarity_target(features: np.ndarray) -> np.ndarray:
    target = np.corrcoef(np.asarray(features, dtype=np.float64))
    target = np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(target, 1.0)
    return target.astype(np.float32)


def cartesian_pair_features(
    lnc_features: np.ndarray,
    disease_features: np.ndarray,
) -> np.ndarray:
    left = np.repeat(np.asarray(lnc_features, dtype=np.float32), len(disease_features), axis=0)
    right = np.tile(np.asarray(disease_features, dtype=np.float32), (len(lnc_features), 1))
    return np.concatenate((left, right), axis=1).astype(np.float32, copy=False)


def gaussian_interaction_profile(profiles: np.ndarray) -> np.ndarray:
    """Compute the Gaussian interaction profile kernel."""
    x = np.asarray(profiles, dtype=np.float64)
    mean_squared_norm = float(np.mean(np.sum(x * x, axis=1)))
    gamma = 1.0 / max(mean_squared_norm, 1e-12)
    squared_norms = np.sum(x * x, axis=1, keepdims=True)
    squared_distances = squared_norms + squared_norms.T - 2.0 * (x @ x.T)
    kernel = np.exp(-gamma * np.maximum(squared_distances, 0.0))
    np.fill_diagonal(kernel, 1.0)
    return kernel.astype(np.float32)
