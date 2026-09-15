"""Custom scikit-learn transformer for the PGA Tour win predictor pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SeasonPercentileTransformer(BaseEstimator, TransformerMixin):
    """Converts raw stat columns into percentile ranks within each season.

    A raw stat value isn't comparable across seasons (driving distance has
    crept up tour-wide over time, for example), so this transformer learns,
    per season and per feature column, the empirical distribution of values
    seen during fit(). At transform time, each value is mapped to its
    percentile rank against its own season's distribution (falling back to
    the pooled distribution across all seasons for a season never seen
    during fit). Missing values are imputed with that season's median first
    (falling back to the overall median).
    """

    def __init__(self, feature_columns, season_column="season"):
        self.feature_columns = feature_columns
        self.season_column = season_column

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.season_distributions_ = {}
        self.overall_distributions_ = {}
        self.season_medians_ = {}
        self.overall_medians_ = {}

        for col in self.feature_columns:
            self.season_distributions_[col] = {}
            self.season_medians_[col] = {}

            overall_values = X[col].dropna().to_numpy(dtype=float)
            self.overall_distributions_[col] = np.sort(overall_values)
            self.overall_medians_[col] = (
                float(np.median(overall_values)) if len(overall_values) else 0.0
            )

            for season, group in X.groupby(self.season_column):
                values = group[col].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                self.season_distributions_[col][season] = np.sort(values)
                self.season_medians_[col][season] = float(np.median(values))

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        output = {}

        for col in self.feature_columns:
            season_dists = self.season_distributions_[col]
            season_meds = self.season_medians_[col]
            overall_dist = self.overall_distributions_[col]
            overall_med = self.overall_medians_[col]

            seasons = X[self.season_column].to_numpy()
            values = X[col].to_numpy(dtype=float)
            percentiles = np.empty(len(X), dtype=float)

            for i in range(len(X)):
                season = seasons[i]
                value = values[i]

                if np.isnan(value):
                    value = season_meds.get(season, overall_med)

                dist = season_dists.get(season)
                if dist is None or len(dist) == 0:
                    dist = overall_dist
                percentiles[i] = np.searchsorted(dist, value, side="right") / len(dist)

            output[col] = percentiles

        return pd.DataFrame(output, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_columns)
