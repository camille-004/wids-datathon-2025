from pathlib import Path

import pandas as pd

from src.models.train import train_model
from src.models.utils import get_latest_artifacts, load_model_artifacts


def load_test_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load test data files.

    Parameters
    ----------
    data_dir : Path
        Path to directory containing test data files

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing test dataframes
    """
    dfs = {
        "categorical": pd.read_excel(
            data_dir / "TEST_CATEGORICAL.xlsx", engine="openpyxl"
        ),
        "connectome": pd.read_csv(
            data_dir / "TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv",
        ),
        "quantitative": pd.read_excel(
            data_dir / "TEST_QUANTITATIVE_METADATA.xlsx", engine="openpyxl"
        ),
    }

    return dfs


def gen_predictions(
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    model_dir: Path | None = None,
) -> pd.DataFrame:
    """Generate predictions for test data.

    Parameters
    ----------
    train_dir : Path
        Path to training data directory
    test_dir : Path
        Path to test data directory
    output_dir : Path
        Path to save predictions
    model_dir : Path | None, optional
        Path to saved model artifacts, by default None

    Returns
    -------
    pd.DataFrame
        Predictions dataframe
    """
    if model_dir and model_dir.exists():
        try:
            model_path, preprocessor_path = get_latest_artifacts(model_dir)
            model, preprocessor = load_model_artifacts(model_path, preprocessor_path)
            print("\nUsing saved model artifacts")

        except FileNotFoundError:
            print("\nNo saved model found, training new model...")
            model, _, preprocessor = train_model(train_dir, save_dir=model_dir)
    else:
        print("\nTraining new model...")
        model, _, preprocessor = train_model(train_dir, save_dir=model_dir)

    test_dfs = load_test_data(test_dir)

    X_test = preprocessor.transform(
        cat_df=test_dfs["categorical"],
        conn_df=test_dfs["connectome"],
        quant_df=test_dfs["quantitative"],
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    submission = pd.DataFrame(
        {
            "participant_id": test_dfs["categorical"]["participant_id"],
            "ADHD_Outcome": predictions[:, 0],
            "Sex_F": predictions[:, 1],
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"\nSubmission saved to {output_dir / 'submission.csv'}")

    adhd_proba, sex_proba = probabilities
    proba_df = pd.DataFrame(
        {
            "participant_id": test_dfs["categorical"]["participant_id"],
            "ADHD_prob": adhd_proba[:, 1],
            "Sex_F_prob": sex_proba[:, 1],
        }
    )
    proba_df.to_csv(output_dir / "probabilities.csv", index=False)
    print(f"Probabilities saved to {output_dir / 'probabilities.csv'}")

    return submission


if __name__ == "__main__":
    base_dir = Path("data")
    train_dir = base_dir / "raw/train"
    test_dir = base_dir / "raw/test"
    output_dir = base_dir / "submissions"
    model_dir = Path("models/saved")

    predictions = gen_predictions(train_dir, test_dir, output_dir, model_dir)

    print("\nSample predictions:")
    print(predictions.head())

    print("\nADHD predictions distribution:")
    print(predictions["ADHD_Outcome"].value_counts())
    print("\nSex predictions distribution:")
    print(predictions["Sex_F"].value_counts())
