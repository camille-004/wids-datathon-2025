import joblib
from datetime import datetime
from pathlib import Path

from src.features.preprocessor import ADHDPreprocessor
from src.models.model import MultiTargetADHD


def save_model_artifacts(
    model: MultiTargetADHD,
    preprocessor: ADHDPreprocessor,
    metrics: dict,
    save_dir: Path,
    prefix: str = "",
) -> tuple[Path, Path]:
    """Save model, preprocessor, and metrics.

    Parameters
    ----------
    model : MultiTargetADHD
        Trained model
    preprocessor : ADHDPreprocessor
        Fitted preprocessor
    metrics : dict
        Model metrics
    save_dir : Path
        Directory to save artifacts
    prefix : str, optional
        Prefix for saved files, by default ""

    Returns
    -------
    tuple[Path, Path]
        Paths to saved model and preprocessor
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{prefix}model_{timestamp}.joblib"
    preprocessor_path = save_dir / f"{prefix}preprocessor_{timestamp}.joblib"
    metrics_path = save_dir / f"{prefix}metrics_{timestamp}.joblib"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(metrics, metrics_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved preprocessor to: {preprocessor_path}")
    print(f"Saved metrics to: {metrics_path}")

    return model_path, preprocessor_path


def load_model_artifacts(
    model_path: Path, preprocessor_path: Path
) -> tuple[MultiTargetADHD, ADHDPreprocessor]:
    """Load model and preprocessor.

    Parameters
    ----------
    model_path : Path
        Path to saved model
    preprocessor_path : Path
        Path to saved preprocessor

    Returns
    -------
    tuple[MultiTargetADHD, ADHDPreprocessor]
        Loaded model and preprocessor
    """
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    print(f"\nLoaded model from: {model_path}")
    print(f"Loaded preprocessor from: {preprocessor_path}")

    return model, preprocessor


def get_latest_artifacts(save_dir: Path, prefix: str = "") -> tuple[Path, Path]:
    """Get paths to latest model and preprocessor.

    Parameters
    ----------
    save_dir : Path
        Directory containing saved artifacts
    prefix : str, optional
        Prefix for saved files, by default ""

    Returns
    -------
    tuple[Path, Path]
        Paths to latest model and preprocessor
    """
    save_dir = Path(save_dir)

    model_files = list(save_dir.glob(f"{prefix}model_*.joblib"))
    preprocessor_files = list(save_dir.glob(f"{prefix}preprocessor_*.joblib"))

    if not model_files or not preprocessor_files:
        raise FileNotFoundError("No saved model artifacts found")

    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    latest_preprocessor = max(preprocessor_files, key=lambda x: x.stat().st_mtime)

    return latest_model, latest_preprocessor
