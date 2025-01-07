import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


class ADHDPreprocessor:
    """Preprocessor for ADHD prediction data.

    Handles preprocessing of categorical, connectome, and quantitative data.
    """

    def __init__(self, num_components_connectome: int = 100) -> None:
        """Initialize preprocessor.

        Parameters
        ----------
        num_components_connectome : int, optional
            Number of PCA components for connectome data, by default 100
        """
        self.num_components = num_components_connectome
        self.scalers: dict[str, StandardScaler] = {}
        self.imputers: dict[str, SimpleImputer] = {}
        self.pca = None
        self.cat_columns = None

    def fit_transform(
        self, cat_df: pd.DataFrame, conn_df: pd.DataFrame, quant_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Fit preprocessor and transform data.

        Parameters
        ----------
        cat_df : pd.DataFrame
            DataFrame with categorical features
        conn_df : pd.DataFrame
            DataFrame with connectome features
        quant_df : pd.DataFrame
            DataFrame with quantitative features

        Returns
        -------
        pd.DataFrame
            Combined preprocessed features DataFrame
        """
        participant_ids = sorted(
            set(cat_df["participant_id"])
            & set(conn_df["participant_id"])
            & set(quant_df["participant_id"])
        )

        print(f"\nNumber of common participants: {len(participant_ids)}")

        cat_df = cat_df[cat_df["participant_id"].isin(participant_ids)].sort_values(
            "participant_id"
        )
        conn_df = conn_df[conn_df["participant_id"].isin(participant_ids)].sort_values(
            "participant_id"
        )
        quant_df = quant_df[
            quant_df["participant_id"].isin(participant_ids)
        ].sort_values("participant_id")

        cat_processed = self._process_categorical(cat_df, fit=True)
        conn_processed = self._process_connectome(conn_df, fit=True)
        quant_processed = self._process_quantitative(quant_df, fit=True)

        result = pd.concat([cat_processed, conn_processed, quant_processed], axis=1)
        print(f"Final processed shape: {result.shape}")

        return result

    def transform(
        self, cat_df: pd.DataFrame, conn_df: pd.DataFrame, quant_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform data using fitted preprocessor.

        Parameters
        ----------
        cat_df : pd.DataFrame
            DataFrame with categorical features
        conn_df : pd.DataFrame
            DataFrame with connectome features
        quant_df : pd.DataFrame
            DataFrame with quantitative features

        Returns
        -------
        pd.DataFrame
            Combined preprocessed features dataframe
        """
        if self.cat_columns is None:
            raise ValueError("Preprocessor must be fitted before transform")

        participant_ids = sorted(
            set(cat_df["participant_id"])
            & set(conn_df["participant_id"])
            & set(quant_df["participant_id"])
        )
        print(f"\nNumber of common participants (transform): {len(participant_ids)}")

        cat_df = cat_df[cat_df["participant_id"].isin(participant_ids)]
        conn_df = conn_df[conn_df["participant_id"].isin(participant_ids)]
        quant_df = quant_df[quant_df["participant_id"].isin(participant_ids)]

        cat_processed = self._process_categorical(cat_df, fit=False)
        conn_processed = self._process_connectome(conn_df, fit=False)
        quant_processed = self._process_quantitative(quant_df, fit=False)

        result = pd.concat([cat_processed, conn_processed, quant_processed], axis=1)
        print(f"Final processed shape (transform): {result.shape}")

        return result

    def _process_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Process categorical features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with categorical features
        fit : bool, optional
            Whether to fit or just transform, by default True

        Returns
        -------
        pd.DataFrame
            Processed categorical features
        """
        processed = pd.get_dummies(df.drop("participant_id", axis=1))

        if fit:
            self.cat_columns = processed.columns
        else:
            missing_cols = set(self.cat_columns) - set(processed.columns)
            for col in missing_cols:
                processed[col] = 0
            processed = processed[self.cat_columns]

        return processed

    def _process_connectome(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Process connectome features using PCA.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with connectome features
        fit : bool, optional
            Whether to fit or just transform, by default True

        Returns
        -------
        pd.DataFrame
            Processed connectome features
        """
        X = df.drop("participant_id", axis=1).values

        if fit:
            self.scalers["connectome"] = StandardScaler()
            self.pca = PCA(n_components=self.num_components)
            X = self.scalers["connectome"].fit_transform(X)
            X = self.pca.fit_transform(X)
        else:
            X = self.scalers["connectome"].transform(X)
            X = self.pca.transform(X)

        return pd.DataFrame(X, columns=[f"pc_{i}" for i in range(X.shape[1])])

    def _process_quantitative(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Process quantitative features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with quantitative features
        fit : bool, optional
            Wehther to fit or just transform, by default True

        Returns
        -------
        pd.DataFrame
            Processed quantitative features
        """
        df_processed = df.drop("participant_id", axis=1)

        if fit:
            self.imputers["quant"] = SimpleImputer(strategy="median")
            self.scalers["quant"] = StandardScaler()

        X = (
            self.imputers["quant"].fit_transform(df_processed)
            if fit
            else self.imputers["quant"].transform(df_processed)
        )
        X = (
            self.scalers["quant"].fit_transform(X)
            if fit
            else self.scalers["quant"].transform(X)
        )

        return pd.DataFrame(X, columns=df_processed.columns)
