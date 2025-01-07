from pathlib import Path

import pandas as pd


def load_train_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all training data files.

    Parameters
    ----------
    data_dir : str | Path
        Path to directory containing the data files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing all loaded dataframes.
    """
    data_dir = Path(data_dir)

    dfs = {
        "categorical": pd.read_excel(
            data_dir / "TRAIN_CATEGORICAL_METADATA.xlsx", engine="openpyxl"
        ),
        "connectome": pd.read_csv(
            data_dir / "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv",
        ),
        "quantitative": pd.read_excel(
            data_dir / "TRAIN_QUANTITATIVE_METADATA.xlsx", engine="openpyxl"
        ),
        "targets": pd.read_excel(
            data_dir / "TRAINING_SOLUTIONS.xlsx", engine="openpyxl"
        ),
    }

    return dfs


def get_feature_groups(dfs: dict[str, pd.DataFrame]) -> dict[str, list]:
    """Extract feature groups from the dataframes.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Dictionary of loaded dataframes

    Returns
    -------
    dict[str, list]
        Dictionary containing feature names grouped by type
    """
    feature_groups = {
        "categorical": dfs["categorical"].columns.tolist(),
        "connectome": dfs["connectome"].columns.tolist(),
        "quantitative": dfs["quantitative"].columns.tolist(),
        "targets": dfs["targets"].columns.tolist(),
    }

    return feature_groups


def get_data_info(dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Get basic information about each dataframe.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Dictionary of loaded dataframes

    Returns
    -------
    dict[str, dict]
        Dictionary containing info about each dataframe
    """
    info = {}
    for name, df in dfs.items():
        info[name] = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().sum(),
            "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
            * 100,
            "dtypes": df.dtypes.value_counts().to_dict(),
        }
    return info
