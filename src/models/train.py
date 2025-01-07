from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.load_data import load_train_data
from src.features.preprocessor import ADHDPreprocessor
from src.models.xgboost_model import XGBoostMultiTarget
from src.models.utils import save_model_artifacts


def create_stratified_folds(y: pd.DataFrame, num_splits: int = 5) -> np.ndarray:
    """Create stratified folds for multi-target data.

    Parameters
    ----------
    y : pd.DataFrame
        Target DataFrame with ADHD_Outcome and Sex_F
    num_splits : int, optional
        Number of folds, by default 5

    Returns
    -------
    np.ndarray
        Array of fold indicies
    """
    combined_target = y["ADHD_Outcome"].astype(str) + "_" + y["Sex_F"].astype(str)

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    return list(skf.split(y, combined_target))


def train_model(
    data_dir: Path,
    num_components: int = 100,
    num_splits: int = 5,
    save_dir: Path | None = None,
) -> tuple[Any, dict[str, float], ADHDPreprocessor]:
    """Train and evaluate model using cross-validation.

    Parameters
    ----------
    data_dir : Path
        Path to data directory
    num_components : int, optional
        Number of PCA components for connectome data, by default 100
    num_splits : int, optional
        Number of cross-validation folds, by default 5

    Returns
    -------
    tuple[Any, dict[str, float], ADHDPreprocessor]
        Tuple of (fitted_model, cv_metrics, fitted_preprocessor)
    """
    dfs = load_train_data(data_dir)

    print("\nInitial data shapes:")
    for name, df in dfs.items():
        print(f"{name}: {df.shape}")

    participant_ids = dfs["targets"]["participant_id"]
    targets = dfs["targets"].set_index("participant_id")
    feature_dfs = {
        key: df.set_index("participant_id")
        for key, df in dfs.items()
        if key != "targets"
    }

    preprocessor = ADHDPreprocessor(num_components_connectome=num_components)
    model = XGBoostMultiTarget(random_state=42, n_estimators=200)

    folds = create_stratified_folds(targets.reset_index(), num_splits=num_splits)

    cv_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\nFold {fold_idx + 1}/{num_splits}")

        train_participants = participant_ids.iloc[train_idx]
        val_participants = participant_ids.iloc[val_idx]

        print(
            f"Train size: {len(train_participants)}, Val size: {len(val_participants)}"
        )

        # Split data
        train_dfs = {
            name: df.loc[train_participants] for name, df in feature_dfs.items()
        }
        val_dfs = {name: df.loc[val_participants] for name, df in feature_dfs.items()}

        print("\nTrain shapes:")
        for name, df in train_dfs.items():
            print(f"{name}: {df.shape}")

        # Preprocess
        X_train = preprocessor.fit_transform(
            cat_df=train_dfs["categorical"].reset_index(),
            conn_df=train_dfs["connectome"].reset_index(),
            quant_df=train_dfs["quantitative"].reset_index(),
        )

        train_targets = targets.loc[train_participants]
        print(f"\nX_train shape: {X_train.shape}")
        print(f"Train targets shape: {train_targets.shape}")

        X_val = preprocessor.transform(
            cat_df=val_dfs["categorical"].reset_index(),
            conn_df=val_dfs["connectome"].reset_index(),
            quant_df=val_dfs["quantitative"].reset_index(),
        )

        val_targets = targets.loc[val_participants]
        print(f"\nX_val shape: {X_val.shape}")
        print(f"Val targets shape: {val_targets.shape}")

        model.fit(X_train, train_targets)
        fold_metrics = model.evaluate(X_val, targets.iloc[val_idx])
        cv_metrics.append(fold_metrics)

        print("Fold metrics:")
        for metric, value in fold_metrics.items():
            print(f"{metric}: {value:.4f}")

    mean_metrics = {}
    for metric in cv_metrics[0].keys():
        values = [m[metric] for m in cv_metrics]
        mean_metrics[metric] = np.mean(values)
        std = np.std(values)
        print(f"\nMean {metric}: {mean_metrics[metric]:.4f} ± {std:.4f}")

    # Fit final model on all data
    X_full = preprocessor.fit_transform(
        cat_df=feature_dfs["categorical"].reset_index(),
        conn_df=feature_dfs["connectome"].reset_index(),
        quant_df=feature_dfs["quantitative"].reset_index(),
    )
    print(f"\nFinal X shape: {X_full.shape}")
    print(f"Final target shape: {targets.shape}")

    model.fit(X_full, targets)

    if save_dir:
        save_model_artifacts(model, preprocessor, mean_metrics, save_dir)

    return model, mean_metrics, preprocessor


if __name__ == "__main__":
    data_dir = Path("data/raw/train")
    save_dir = Path("models/saved")

    model, metrics, preprocessor = train_model(
        data_dir, num_components=100, num_splits=5, save_dir=save_dir
    )

    importance = model.get_feature_importance()
    print("\nTop 10 features for ADHD prediction:")
    print(importance["adhd"].head(10))
    print("\nTop 10 features for sex prediction:")
    print(importance["sex"].head(10))
