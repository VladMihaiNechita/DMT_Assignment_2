"""Raw Expedia preprocessing for the packed transformer ranker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

try:
    from .features import (
        COMPETITOR_COLUMNS,
        HISTORICAL_PRIOR_KEYS,
        MARKET_AGG_FEATURE_COLUMNS,
        NORMALIZE_COLUMNS,
        NONRANDOM_PRIOR_COLUMNS,
        PRIOR_COLUMNS,
        RESULT_BASE_COLUMNS,
        RESULT_CATEGORICAL_COLUMNS,
        SEARCH_RELATIVE_BASE_COLUMNS,
        SEARCH_BASE_COLUMNS,
        SEARCH_CATEGORICAL_COLUMNS,
        SEARCH_NUMERIC_COLUMNS,
        FeatureSpec,
        get_categorical_columns,
        get_prior_relative_columns,
        get_result_numeric_columns,
    )
except ImportError:  # pragma: no cover - supports running from inside Code/
    from features import (
        COMPETITOR_COLUMNS,
        HISTORICAL_PRIOR_KEYS,
        MARKET_AGG_FEATURE_COLUMNS,
        NORMALIZE_COLUMNS,
        NONRANDOM_PRIOR_COLUMNS,
        PRIOR_COLUMNS,
        RESULT_BASE_COLUMNS,
        RESULT_CATEGORICAL_COLUMNS,
        SEARCH_RELATIVE_BASE_COLUMNS,
        SEARCH_BASE_COLUMNS,
        SEARCH_CATEGORICAL_COLUMNS,
        SEARCH_NUMERIC_COLUMNS,
        FeatureSpec,
        get_categorical_columns,
        get_prior_relative_columns,
        get_result_numeric_columns,
    )


def _ensure_columns(df: pd.DataFrame, columns: list[str], fill_value: Any = np.nan) -> None:
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value


def _add_time_features(df: pd.DataFrame) -> None:
    if "date_time" not in df.columns:
        df["month_of_year"] = np.nan
        df["day_of_week"] = np.nan
        df["week_of_year"] = np.nan
        df["hour_of_day"] = np.nan
        df["hour_bucket"] = np.nan
        df["time_epoch"] = np.nan
        return

    dt = pd.to_datetime(df["date_time"], errors="coerce")
    df["month_of_year"] = dt.dt.month.astype("float32")
    df["day_of_week"] = dt.dt.dayofweek.astype("float32")
    df["week_of_year"] = dt.dt.isocalendar().week.astype("float32")
    df["hour_of_day"] = dt.dt.hour.astype("float32")
    df["hour_bucket"] = (dt.dt.hour // 6).astype("float32")

    epoch = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = dt.notna()
    if valid.any():
        epoch.loc[valid] = dt.loc[valid].astype("int64").to_numpy(dtype="float64") / 1_000_000_000.0
    df["time_epoch"] = epoch.astype("float32")


def _add_required_flags(df: pd.DataFrame) -> None:
    _ensure_columns(
        df,
        [
            "visitor_hist_starrating",
            "visitor_hist_adr_usd",
            "prop_starrating",
            "prop_review_score",
            "prop_location_score2",
            "prop_log_historical_price",
            "srch_query_affinity_score",
            "orig_destination_distance",
        ],
    )

    df["is_NULL_visitor_hist_starrating"] = df["visitor_hist_starrating"].isna().astype("int8")
    df["is_NULL_visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].isna().astype("int8")
    df["is_0_prop_starrating"] = (df["prop_starrating"] == 0).astype("int8")
    df["is_NULL_prop_review_score"] = df["prop_review_score"].isna().astype("int8")
    df["is_0_prop_review_score"] = (df["prop_review_score"] == 0).astype("int8")
    df["is_NULL_prop_location_score2"] = df["prop_location_score2"].isna().astype("int8")
    df["is_0_prop_log_historical_price"] = (df["prop_log_historical_price"] == 0).astype("int8")
    df["is_NULL_srch_query_affinity_score"] = df["srch_query_affinity_score"].isna().astype("int8")
    df["is_NULL_orig_destination_distance"] = df["orig_destination_distance"].isna().astype("int8")

    for i in range(1, 9):
        for suffix in ("rate", "inv", "rate_percent_diff"):
            col = f"comp{i}_{suffix}"
            if col not in df.columns:
                df[col] = np.nan
            df[f"is_NULL_comp{i}_{suffix}"] = df[col].isna().astype("int8")


def make_relevance_labels(df: pd.DataFrame) -> np.ndarray:
    """Competition relevance: booking=5, click-only=1, else 0."""

    labels = np.zeros(len(df), dtype="float32")
    if "click_bool" in df.columns:
        labels[df["click_bool"].fillna(0).astype("int8").to_numpy() == 1] = 1.0
    if "booking_bool" in df.columns:
        labels[df["booking_bool"].fillna(0).astype("int8").to_numpy() == 1] = 5.0
    return labels


def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / denom.replace(0, np.nan)


def _add_query_size(df: pd.DataFrame) -> None:
    if "srch_id" in df.columns:
        df["n_props"] = df.groupby("srch_id", sort=False)["prop_id"].transform("size").astype("float32")
    else:
        df["n_props"] = np.nan


def _add_price_features(df: pd.DataFrame) -> None:
    price = df["price_usd"].clip(lower=0)
    nights = df["srch_length_of_stay"].replace(0, np.nan)
    rooms = df["srch_room_count"].replace(0, np.nan)
    adults = df["srch_adults_count"].replace(0, np.nan)
    persons = (df["srch_adults_count"].fillna(0) + df["srch_children_count"].fillna(0)).replace(0, np.nan)

    df["log_price"] = np.log1p(price)
    df["price_per_night"] = price / nights
    df["price_per_room"] = price / rooms
    df["price_per_adult"] = price / adults
    df["price_per_person"] = price / persons
    df["total_search_size"] = rooms * nights

    historical_price = np.exp(df["prop_log_historical_price"])
    df["price_vs_historical"] = df["log_price"] - df["prop_log_historical_price"]
    df["ump"] = historical_price - price
    df["price_ratio_historical"] = _safe_divide(price, historical_price)


def _add_match_features(df: pd.DataFrame) -> None:
    df["star_gap"] = (df["prop_starrating"] - df["visitor_hist_starrating"]).abs()
    df["adr_gap"] = (df["price_usd"] - df["visitor_hist_adr_usd"]).abs()
    df["domestic"] = (df["visitor_location_country_id"] == df["prop_country_id"]).astype("int8")


def _add_competitor_aggregates(df: pd.DataFrame) -> None:
    rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
    inv_cols = [f"comp{i}_inv" for i in range(1, 9)]
    pct_cols = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]

    df["comp_rate_mean"] = df[rate_cols].mean(axis=1)
    df["comp_n_competitors"] = df[rate_cols].notna().sum(axis=1)
    df["comp_n_better"] = (df[rate_cols] == 1).sum(axis=1)
    df["comp_n_worse"] = (df[rate_cols] == -1).sum(axis=1)
    df["comp_inv_mean"] = df[inv_cols].mean(axis=1)
    df["comp_pct_mean"] = df[pct_cols].mean(axis=1)
    df["comp_pct_max"] = df[pct_cols].max(axis=1)
    df["comp_pct_min"] = df[pct_cols].min(axis=1)


def _add_composite_features(df: pd.DataFrame) -> None:
    df["star_x_review"] = df["prop_starrating"] * df["prop_review_score"]
    df["loc_score1_x_score2"] = df["prop_location_score1"] * df["prop_location_score2"]


def _encode_ord(values: pd.Series, scale: float, max_code: int) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float32")
    out = np.zeros(len(arr), dtype="int64")
    valid = np.isfinite(arr)
    out[valid] = np.clip(np.rint(arr[valid] * scale).astype("int64"), 0, max_code) + 1
    return out


def _encode_log_bucket(values: pd.Series, scale: float, max_code: int) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float32")
    out = np.zeros(len(arr), dtype="int64")
    valid = np.isfinite(arr)
    if valid.any():
        bucket = np.floor(np.log1p(np.clip(arr[valid], 0.0, None)) * scale).astype("int64")
        out[valid] = np.clip(bucket, 0, max_code) + 1
    return out


def _encode_pct_bucket(values: pd.Series, bucket_count: int) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float32")
    out = np.zeros(len(arr), dtype="int64")
    valid = np.isfinite(arr)
    if valid.any():
        bucket = np.ceil(np.clip(arr[valid], 0.0, 1.0) * bucket_count).astype("int64")
        out[valid] = np.clip(bucket, 1, bucket_count)
    return out


def _add_ordinal_category_features(df: pd.DataFrame) -> None:
    df["prop_starrating_ord"] = _encode_ord(df["prop_starrating"], scale=1.0, max_code=5)
    df["prop_review_score_ord"] = _encode_ord(df["prop_review_score"], scale=2.0, max_code=10)
    for i in range(1, 9):
        rate = pd.to_numeric(df[f"comp{i}_rate"], errors="coerce").to_numpy(dtype="float32")
        rate_out = np.zeros(len(rate), dtype="int64")
        valid_rate = np.isfinite(rate)
        rate_out[valid_rate] = np.clip(rate[valid_rate].astype("int64") + 2, 1, 3)
        df[f"comp{i}_rate_ord"] = rate_out

        inv = pd.to_numeric(df[f"comp{i}_inv"], errors="coerce").to_numpy(dtype="float32")
        inv_out = np.zeros(len(inv), dtype="int64")
        valid_inv = np.isfinite(inv)
        inv_out[valid_inv] = np.clip(inv[valid_inv].astype("int64") + 1, 1, 2)
        df[f"comp{i}_inv_ord"] = inv_out


def _add_bucket_category_features(df: pd.DataFrame) -> None:
    df["price_bucket_ord"] = _encode_log_bucket(df["price_usd"], scale=8.0, max_code=80)
    df["booking_window_bucket_ord"] = _encode_log_bucket(df["srch_booking_window"], scale=8.0, max_code=64)
    df["stay_length_bucket_ord"] = _encode_ord(df["srch_length_of_stay"], scale=1.0, max_code=30)
    df["price_rank_bucket_ord"] = _encode_pct_bucket(df["price_usd_rank_pct"], bucket_count=20)
    df["price_diff_rank_bucket_ord"] = _encode_pct_bucket(df["price_vs_historical_rank_pct"], bucket_count=20)
    df["star_rank_bucket_ord"] = _encode_pct_bucket(df["prop_starrating_rank_pct"], bucket_count=10)
    df["loc2_rank_bucket_ord"] = _encode_pct_bucket(df["prop_location_score2_rank_pct"], bucket_count=20)


def _add_search_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    if "srch_id" not in df.columns:
        return df
    groups = df.groupby("srch_id", sort=False)
    new_cols: dict[str, pd.Series] = {}
    for col in SEARCH_RELATIVE_BASE_COLUMNS:
        if col not in df.columns:
            continue
        values = df[col].astype("float32")
        g_mean = groups[col].transform("mean")
        g_min = groups[col].transform("min")
        g_max = groups[col].transform("max")
        g_std = groups[col].transform("std").replace(0, np.nan)
        new_cols[f"{col}_diff_mean"] = values - g_mean
        new_cols[f"{col}_diff_min"] = values - g_min
        new_cols[f"{col}_zscore"] = (values - g_mean) / g_std
        new_cols[f"{col}_range_pos"] = (values - g_min) / (g_max - g_min).replace(0, np.nan)
        new_cols[f"{col}_rank_pct"] = groups[col].rank(method="average", pct=True)
    if new_cols:
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).copy()
    return df


@dataclass
class ExpediaPreprocessor:
    """Fit-on-train preprocessing with deterministic test transforms."""

    use_market_aggregates: bool = True
    use_bucket_categories: bool = True
    use_search_relative: bool = True
    use_prior_relative: bool = True
    prior_relative_mode: str = "all"
    normalize_columns: list[str] = field(default_factory=lambda: list(NORMALIZE_COLUMNS))
    category_values: dict[str, np.ndarray] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    feature_spec: FeatureSpec | None = None
    prior_alpha: float = 50.0
    global_booking: float = 0.0
    global_click: float = 0.0
    global_relevance: float = 0.0
    nonrandom_global_booking: float = 0.0
    nonrandom_global_click: float = 0.0
    nonrandom_global_relevance: float = 0.0
    position_global_mean: float = 0.0
    position_global_median: float = 0.0
    position_stats: pd.DataFrame | None = None
    nonrandom_prop_stats: pd.DataFrame | None = None
    historical_prior_stats: dict[tuple[str, ...], pd.DataFrame] = field(default_factory=dict)
    market_aggregate_stats: dict[str, pd.DataFrame] = field(default_factory=dict)
    market_global_values: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "ExpediaPreprocessor":
        frame = self._base_transform(df)
        self._fit_from_base_frame(frame)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_spec is None:
            raise RuntimeError("ExpediaPreprocessor must be fit before transform().")
        frame = self._base_transform(df)
        frame = self._add_fitted_aggregate_features(frame, is_train=False)
        return self._apply_fitted_transforms(frame)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = self._base_transform(df)
        frame = self._fit_from_base_frame(frame)
        return self._apply_fitted_transforms(frame)

    def _fit_from_base_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._fit_position_stats(frame)
        if getattr(self, "use_market_aggregates", True):
            self._fit_market_aggregate_stats(frame)
        self._fit_prior_stats(frame)

        categorical_columns = self._categorical_columns()
        result_numeric_columns = self._result_numeric_columns()
        self.category_values = {}
        for col in categorical_columns:
            values = frame[col].fillna(0).astype("int64").to_numpy()
            self.category_values[col] = np.sort(np.unique(values))

        self.feature_spec = FeatureSpec(
            search_numeric=list(SEARCH_NUMERIC_COLUMNS),
            result_numeric=result_numeric_columns,
            categorical=categorical_columns,
            categorical_cardinalities={
                col: int(len(values) + 2)
                for col, values in self.category_values.items()
            },
        )

        enriched = self._add_fitted_aggregate_features(frame, is_train=True)
        _ensure_columns(enriched, self.feature_spec.search_numeric + self.feature_spec.result_numeric)
        self.means = {}
        self.stds = {}
        for col in self.normalize_columns:
            if col not in enriched.columns:
                continue
            values = pd.to_numeric(enriched[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if not np.isfinite(mean):
                mean = 0.0
            self.means[col] = mean
            self.stds[col] = std if np.isfinite(std) and std > 1e-6 else 1.0
        return enriched

    def _apply_fitted_transforms(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.feature_spec is None:
            raise RuntimeError("ExpediaPreprocessor must be fit before transform().")
        _ensure_columns(frame, self.feature_spec.search_numeric + self.feature_spec.result_numeric)

        for col, mean in self.means.items():
            std = self.stds[col]
            values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            frame[col] = ((values.astype("float32") - mean) / std).astype("float32")

        for col, known_values in self.category_values.items():
            raw = frame[col].fillna(0).astype("int64").to_numpy()
            frame[col] = self._map_category(raw, known_values)

        for col in self.feature_spec.search_numeric + self.feature_spec.result_numeric:
            values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            frame[col] = values.fillna(0.0).astype("float32")
        for col in self.feature_spec.categorical:
            frame[col] = frame[col].astype("int64")

        return frame

    def _fit_position_stats(self, frame: pd.DataFrame) -> None:
        if "position" not in frame.columns or frame["position"].dropna().empty:
            self.position_stats = None
            self.position_global_mean = 0.0
            self.position_global_median = 0.0
            return

        ranked = frame.loc[frame["random_bool"].fillna(0) != 1]
        if ranked.empty:
            ranked = frame
        self.position_global_mean = float(ranked["position"].mean())
        self.position_global_median = float(ranked["position"].median())
        self.position_stats = ranked.groupby("prop_id", sort=False)["position"].agg(
            prop_mean_position="mean",
            prop_median_position="median",
            prop_position_count="size",
        )

    def _fit_market_aggregate_stats(self, frame: pd.DataFrame) -> None:
        work = frame.copy(deep=False)
        for col in (
            "price_usd",
            "log_price",
            "promotion_flag",
            "prop_starrating",
            "prop_review_score",
            "prop_location_score1",
        ):
            work[col] = pd.to_numeric(work[col], errors="coerce").astype("float32")

        self.market_global_values = {
            "price_mean": float(work["price_usd"].mean()),
            "price_std": float(work["price_usd"].std(ddof=0)),
            "log_price_mean": float(work["log_price"].mean()),
            "promotion_rate": float(work["promotion_flag"].mean()),
            "star_mean": float(work["prop_starrating"].mean()),
            "review_mean": float(work["prop_review_score"].mean()),
            "loc_score1_mean": float(work["prop_location_score1"].mean()),
        }
        for key, value in list(self.market_global_values.items()):
            if not np.isfinite(value):
                self.market_global_values[key] = 0.0
        if self.market_global_values["price_std"] <= 1e-6:
            self.market_global_values["price_std"] = 1.0

        self.market_aggregate_stats = {}
        if "prop_id" in work.columns:
            prop_stats = work.groupby("prop_id", sort=False).agg(
                prop_price_mean=("price_usd", "mean"),
                prop_price_std=("price_usd", "std"),
                prop_price_count_log=("price_usd", "size"),
                prop_log_price_mean=("log_price", "mean"),
                prop_promotion_rate=("promotion_flag", "mean"),
            )
            prop_stats["prop_price_std"] = prop_stats["prop_price_std"].fillna(self.market_global_values["price_std"])
            prop_stats["prop_price_count_log"] = np.log1p(prop_stats["prop_price_count_log"])
            self.market_aggregate_stats["prop"] = prop_stats

        if {"prop_id", "srch_destination_id"}.issubset(work.columns):
            prop_dest_stats = work.groupby(["prop_id", "srch_destination_id"], sort=False).agg(
                prop_dest_price_mean=("price_usd", "mean"),
                prop_dest_price_std=("price_usd", "std"),
                prop_dest_count_log=("price_usd", "size"),
                prop_dest_log_price_mean=("log_price", "mean"),
            )
            prop_dest_stats["prop_dest_price_std"] = prop_dest_stats["prop_dest_price_std"].fillna(
                self.market_global_values["price_std"]
            )
            prop_dest_stats["prop_dest_count_log"] = np.log1p(prop_dest_stats["prop_dest_count_log"])
            self.market_aggregate_stats["prop_dest"] = prop_dest_stats

        if "srch_destination_id" in work.columns:
            dest_stats = work.groupby("srch_destination_id", sort=False).agg(
                dest_price_mean=("price_usd", "mean"),
                dest_price_std=("price_usd", "std"),
                dest_count_log=("price_usd", "size"),
                dest_log_price_mean=("log_price", "mean"),
                dest_promotion_rate=("promotion_flag", "mean"),
                dest_star_mean=("prop_starrating", "mean"),
                dest_review_mean=("prop_review_score", "mean"),
                dest_loc_score1_mean=("prop_location_score1", "mean"),
            )
            dest_stats["dest_price_std"] = dest_stats["dest_price_std"].fillna(self.market_global_values["price_std"])
            dest_stats["dest_count_log"] = np.log1p(dest_stats["dest_count_log"])
            self.market_aggregate_stats["dest"] = dest_stats

    def _fit_prior_stats(self, frame: pd.DataFrame) -> None:
        work = frame.copy(deep=False)
        work["relevance"] = make_relevance_labels(work)
        self.global_booking = float(work["booking_bool"].fillna(0).mean())
        self.global_click = float(work["click_bool"].fillna(0).mean())
        self.global_relevance = float(work["relevance"].mean())

        self.historical_prior_stats = {}
        for keys in HISTORICAL_PRIOR_KEYS:
            if not all(col in work.columns for col in keys):
                continue
            prefix = "_x_".join(keys)
            grouped = work.groupby(list(keys), sort=False, dropna=False)
            stats = grouped.agg(
                booking_sum=("booking_bool", "sum"),
                click_sum=("click_bool", "sum"),
                relevance_sum=("relevance", "sum"),
                impressions=("relevance", "size"),
            )
            denom = stats["impressions"] + self.prior_alpha
            stats[f"{prefix}_booking_rate"] = (stats["booking_sum"] + self.prior_alpha * self.global_booking) / denom
            stats[f"{prefix}_click_rate"] = (stats["click_sum"] + self.prior_alpha * self.global_click) / denom
            stats[f"{prefix}_relevance_mean"] = (stats["relevance_sum"] + self.prior_alpha * self.global_relevance) / denom
            stats[f"{prefix}_impressions_log"] = np.log1p(stats["impressions"])
            self.historical_prior_stats[keys] = stats[
                [
                    f"{prefix}_booking_rate",
                    f"{prefix}_click_rate",
                    f"{prefix}_relevance_mean",
                    f"{prefix}_impressions_log",
                ]
            ]

        if "random_bool" not in work.columns:
            self.nonrandom_prop_stats = None
            return
        nonrandom = work.loc[work["random_bool"].fillna(0) == 0]
        if nonrandom.empty:
            self.nonrandom_prop_stats = None
            self.nonrandom_global_booking = self.global_booking
            self.nonrandom_global_click = self.global_click
            self.nonrandom_global_relevance = self.global_relevance
            return

        self.nonrandom_global_booking = float(nonrandom["booking_bool"].fillna(0).mean())
        self.nonrandom_global_click = float(nonrandom["click_bool"].fillna(0).mean())
        self.nonrandom_global_relevance = float(make_relevance_labels(nonrandom).mean())
        grouped = nonrandom.groupby("prop_id", sort=False)
        stats = grouped.agg(
            booking_sum=("booking_bool", "sum"),
            click_sum=("click_bool", "sum"),
            relevance_sum=("relevance", "sum"),
            impressions=("relevance", "size"),
        )
        denom = stats["impressions"] + self.prior_alpha
        stats["prop_id_booking_rate_nonrandom"] = (
            stats["booking_sum"] + self.prior_alpha * self.nonrandom_global_booking
        ) / denom
        stats["prop_id_click_rate_nonrandom"] = (
            stats["click_sum"] + self.prior_alpha * self.nonrandom_global_click
        ) / denom
        stats["prop_id_relevance_mean_nonrandom"] = (
            stats["relevance_sum"] + self.prior_alpha * self.nonrandom_global_relevance
        ) / denom
        stats["prop_id_impressions_log_nonrandom"] = np.log1p(stats["impressions"])
        self.nonrandom_prop_stats = stats

    def _add_fitted_aggregate_features(self, frame: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        if getattr(self, "use_market_aggregates", True):
            self._add_market_aggregate_features(frame)
        self._add_position_features(frame)
        self._add_historical_prior_features(frame, is_train=is_train)
        self._add_nonrandom_prior_features(frame, is_train=is_train)
        if getattr(self, "use_prior_relative", True):
            return self._add_prior_relative_features(frame)
        return frame

    def _add_market_aggregate_features(self, frame: pd.DataFrame) -> None:
        for col in MARKET_AGG_FEATURE_COLUMNS:
            if col in frame.columns:
                frame.drop(columns=col, inplace=True)

        defaults = {
            "prop_price_mean": self.market_global_values.get("price_mean", 0.0),
            "prop_price_std": self.market_global_values.get("price_std", 1.0),
            "prop_price_count_log": 0.0,
            "prop_log_price_mean": self.market_global_values.get("log_price_mean", 0.0),
            "prop_promotion_rate": self.market_global_values.get("promotion_rate", 0.0),
            "prop_dest_price_mean": self.market_global_values.get("price_mean", 0.0),
            "prop_dest_price_std": self.market_global_values.get("price_std", 1.0),
            "prop_dest_count_log": 0.0,
            "prop_dest_log_price_mean": self.market_global_values.get("log_price_mean", 0.0),
            "dest_price_mean": self.market_global_values.get("price_mean", 0.0),
            "dest_price_std": self.market_global_values.get("price_std", 1.0),
            "dest_count_log": 0.0,
            "dest_log_price_mean": self.market_global_values.get("log_price_mean", 0.0),
            "dest_promotion_rate": self.market_global_values.get("promotion_rate", 0.0),
            "dest_star_mean": self.market_global_values.get("star_mean", 0.0),
            "dest_review_mean": self.market_global_values.get("review_mean", 0.0),
            "dest_loc_score1_mean": self.market_global_values.get("loc_score1_mean", 0.0),
        }

        for col, value in defaults.items():
            frame[col] = value

        prop_stats = self.market_aggregate_stats.get("prop")
        if prop_stats is not None and "prop_id" in frame.columns:
            for col in prop_stats.columns:
                frame[col] = frame["prop_id"].map(prop_stats[col]).fillna(defaults.get(col, 0.0))

        prop_dest_stats = self.market_aggregate_stats.get("prop_dest")
        if prop_dest_stats is not None and {"prop_id", "srch_destination_id"}.issubset(frame.columns):
            index = pd.MultiIndex.from_frame(frame[["prop_id", "srch_destination_id"]])
            for col in prop_dest_stats.columns:
                values = prop_dest_stats[col].reindex(index).to_numpy()
                frame[col] = pd.Series(values, index=frame.index).fillna(defaults.get(col, 0.0))

        dest_stats = self.market_aggregate_stats.get("dest")
        if dest_stats is not None and "srch_destination_id" in frame.columns:
            for col in dest_stats.columns:
                frame[col] = frame["srch_destination_id"].map(dest_stats[col]).fillna(defaults.get(col, 0.0))

        price = pd.to_numeric(frame["price_usd"], errors="coerce").astype("float32")
        log_price = pd.to_numeric(frame["log_price"], errors="coerce").astype("float32")
        promotion = pd.to_numeric(frame["promotion_flag"], errors="coerce").astype("float32")
        star = pd.to_numeric(frame["prop_starrating"], errors="coerce").astype("float32")
        review = pd.to_numeric(frame["prop_review_score"], errors="coerce").astype("float32")
        loc_score1 = pd.to_numeric(frame["prop_location_score1"], errors="coerce").astype("float32")

        frame["price_vs_prop_mean"] = price - frame["prop_price_mean"]
        frame["log_price_vs_prop_mean"] = log_price - frame["prop_log_price_mean"]
        frame["promotion_vs_prop_rate"] = promotion - frame["prop_promotion_rate"]
        frame["price_vs_prop_dest_mean"] = price - frame["prop_dest_price_mean"]
        frame["log_price_vs_prop_dest_mean"] = log_price - frame["prop_dest_log_price_mean"]
        frame["price_vs_dest_mean"] = price - frame["dest_price_mean"]
        frame["log_price_vs_dest_mean"] = log_price - frame["dest_log_price_mean"]
        frame["promotion_vs_dest_rate"] = promotion - frame["dest_promotion_rate"]
        frame["star_vs_dest_mean"] = star - frame["dest_star_mean"]
        frame["review_vs_dest_mean"] = review - frame["dest_review_mean"]
        frame["loc_score1_vs_dest_mean"] = loc_score1 - frame["dest_loc_score1_mean"]

    def _add_position_features(self, frame: pd.DataFrame) -> None:
        for col in ("prop_mean_position", "prop_median_position", "prop_position_count"):
            if col in frame.columns:
                frame.drop(columns=col, inplace=True)
        if self.position_stats is not None and not self.position_stats.empty:
            frame["prop_mean_position"] = frame["prop_id"].map(self.position_stats["prop_mean_position"])
            frame["prop_median_position"] = frame["prop_id"].map(self.position_stats["prop_median_position"])
            frame["prop_position_count"] = frame["prop_id"].map(self.position_stats["prop_position_count"])
        else:
            frame["prop_mean_position"] = np.nan
            frame["prop_median_position"] = np.nan
            frame["prop_position_count"] = np.nan
        frame["prop_mean_position"] = frame["prop_mean_position"].fillna(self.position_global_mean)
        frame["prop_median_position"] = frame["prop_median_position"].fillna(self.position_global_median)
        frame["prop_position_count"] = frame["prop_position_count"].fillna(0.0)

    def _add_historical_prior_features(self, frame: pd.DataFrame, is_train: bool) -> None:
        if is_train and {"booking_bool", "click_bool"}.issubset(frame.columns):
            work = frame.copy(deep=False)
            work["relevance"] = make_relevance_labels(work)
            for keys in HISTORICAL_PRIOR_KEYS:
                if not all(col in work.columns for col in keys):
                    continue
                prefix = "_x_".join(keys)
                grouped = work.groupby(list(keys), sort=False, dropna=False)
                count = grouped["relevance"].transform("size").astype("float32")
                booking_sum = grouped["booking_bool"].transform("sum").astype("float32")
                click_sum = grouped["click_bool"].transform("sum").astype("float32")
                relevance_sum = grouped["relevance"].transform("sum").astype("float32")
                loo_count = (count - 1.0).clip(lower=0.0)
                denom = loo_count + self.prior_alpha
                frame[f"{prefix}_booking_rate"] = (
                    booking_sum - work["booking_bool"].fillna(0) + self.prior_alpha * self.global_booking
                ) / denom
                frame[f"{prefix}_click_rate"] = (
                    click_sum - work["click_bool"].fillna(0) + self.prior_alpha * self.global_click
                ) / denom
                frame[f"{prefix}_relevance_mean"] = (
                    relevance_sum - work["relevance"] + self.prior_alpha * self.global_relevance
                ) / denom
                frame[f"{prefix}_impressions_log"] = np.log1p(loo_count)
            return

        for keys, stats in self.historical_prior_stats.items():
            prefix = "_x_".join(keys)
            cols = [
                f"{prefix}_booking_rate",
                f"{prefix}_click_rate",
                f"{prefix}_relevance_mean",
                f"{prefix}_impressions_log",
            ]
            for col in cols:
                if col in frame.columns:
                    frame.drop(columns=col, inplace=True)
            if all(col in frame.columns for col in keys):
                frame[cols] = frame.merge(stats, how="left", left_on=list(keys), right_index=True)[cols]
            else:
                for col in cols:
                    frame[col] = np.nan
            frame[f"{prefix}_booking_rate"] = frame[f"{prefix}_booking_rate"].fillna(self.global_booking)
            frame[f"{prefix}_click_rate"] = frame[f"{prefix}_click_rate"].fillna(self.global_click)
            frame[f"{prefix}_relevance_mean"] = frame[f"{prefix}_relevance_mean"].fillna(self.global_relevance)
            frame[f"{prefix}_impressions_log"] = frame[f"{prefix}_impressions_log"].fillna(0.0)

    def _add_nonrandom_prior_features(self, frame: pd.DataFrame, is_train: bool) -> None:
        if is_train and self.nonrandom_prop_stats is not None and {"booking_bool", "click_bool"}.issubset(frame.columns):
            stats = self.nonrandom_prop_stats
            booking_sum = frame["prop_id"].map(stats["booking_sum"]).fillna(0.0).to_numpy()
            click_sum = frame["prop_id"].map(stats["click_sum"]).fillna(0.0).to_numpy()
            relevance_sum = frame["prop_id"].map(stats["relevance_sum"]).fillna(0.0).to_numpy()
            impressions = frame["prop_id"].map(stats["impressions"]).fillna(0.0).to_numpy()
            nonrandom_mask = (frame["random_bool"].fillna(0).to_numpy() == 0)
            relevance = make_relevance_labels(frame)
            loo_impressions = np.clip(impressions - nonrandom_mask.astype("float64"), 0.0, None)
            denom = loo_impressions + self.prior_alpha
            frame["prop_id_booking_rate_nonrandom"] = (
                booking_sum - frame["booking_bool"].fillna(0).to_numpy() * nonrandom_mask
                + self.prior_alpha * self.nonrandom_global_booking
            ) / denom
            frame["prop_id_click_rate_nonrandom"] = (
                click_sum - frame["click_bool"].fillna(0).to_numpy() * nonrandom_mask
                + self.prior_alpha * self.nonrandom_global_click
            ) / denom
            frame["prop_id_relevance_mean_nonrandom"] = (
                relevance_sum - relevance * nonrandom_mask
                + self.prior_alpha * self.nonrandom_global_relevance
            ) / denom
            frame["prop_id_impressions_log_nonrandom"] = np.log1p(loo_impressions)
            return

        for col in NONRANDOM_PRIOR_COLUMNS:
            if col in frame.columns:
                frame.drop(columns=col, inplace=True)
        if self.nonrandom_prop_stats is not None and not self.nonrandom_prop_stats.empty:
            stats = self.nonrandom_prop_stats
            for col in NONRANDOM_PRIOR_COLUMNS:
                frame[col] = frame["prop_id"].map(stats[col])
        else:
            for col in NONRANDOM_PRIOR_COLUMNS:
                frame[col] = np.nan
        frame["prop_id_booking_rate_nonrandom"] = frame["prop_id_booking_rate_nonrandom"].fillna(self.nonrandom_global_booking)
        frame["prop_id_click_rate_nonrandom"] = frame["prop_id_click_rate_nonrandom"].fillna(self.nonrandom_global_click)
        frame["prop_id_relevance_mean_nonrandom"] = frame["prop_id_relevance_mean_nonrandom"].fillna(self.nonrandom_global_relevance)
        frame["prop_id_impressions_log_nonrandom"] = frame["prop_id_impressions_log_nonrandom"].fillna(0.0)

    def _add_prior_relative_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        mode = getattr(self, "prior_relative_mode", "all")
        if not getattr(self, "use_prior_relative", True):
            mode = "none"
        if "srch_id" not in frame.columns or mode == "none":
            return frame
        if mode == "all":
            selected_relative = None
        else:
            selected_relative = set(get_prior_relative_columns(mode))

        groups = frame.groupby("srch_id", sort=False)
        new_cols: dict[str, pd.Series] = {}
        for col in PRIOR_COLUMNS:
            if col not in frame.columns:
                continue
            diff_col = f"{col}_diff_mean"
            rank_col = f"{col}_rank_pct"
            if selected_relative is not None and diff_col not in selected_relative and rank_col not in selected_relative:
                continue
            values = frame[col].astype("float32")
            if selected_relative is None or diff_col in selected_relative:
                new_cols[diff_col] = values - groups[col].transform("mean")
            if selected_relative is None or rank_col in selected_relative:
                new_cols[rank_col] = groups[col].rank(method="average", pct=True)
        if new_cols:
            return pd.concat([frame, pd.DataFrame(new_cols, index=frame.index)], axis=1).copy()
        return frame

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "ExpediaPreprocessor":
        with Path(path).open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _map_category(raw: np.ndarray, known_values: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(known_values, raw)
        valid = positions < len(known_values)
        mapped = np.ones(raw.shape[0], dtype="int64")
        if valid.any():
            valid_idx = np.flatnonzero(valid)
            exact = known_values[positions[valid]] == raw[valid]
            mapped[valid_idx[exact]] = positions[valid][exact] + 2
        return mapped

    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        if "prop_id" in frame.columns:
            frame["_submission_prop_id"] = (
                pd.to_numeric(frame["prop_id"], errors="coerce")
                .fillna(-1)
                .astype("int64")
            )
        _add_time_features(frame)
        _ensure_columns(
            frame,
            SEARCH_CATEGORICAL_COLUMNS
            + RESULT_CATEGORICAL_COLUMNS
            + SEARCH_BASE_COLUMNS
            + RESULT_BASE_COLUMNS
            + COMPETITOR_COLUMNS,
        )
        _add_required_flags(frame)
        _add_query_size(frame)
        _add_price_features(frame)
        _add_match_features(frame)
        _add_competitor_aggregates(frame)
        _add_composite_features(frame)
        _add_ordinal_category_features(frame)
        if getattr(self, "use_search_relative", True) or getattr(self, "use_bucket_categories", True):
            frame = _add_search_relative_features(frame)
        if getattr(self, "use_bucket_categories", True):
            _add_bucket_category_features(frame)

        model_columns = SEARCH_CATEGORICAL_COLUMNS + RESULT_CATEGORICAL_COLUMNS
        frame[model_columns] = frame[model_columns].fillna(0)
        return frame

    def _categorical_columns(self) -> list[str]:
        return get_categorical_columns(getattr(self, "use_bucket_categories", True))

    def _result_numeric_columns(self) -> list[str]:
        prior_relative_mode = getattr(self, "prior_relative_mode", "all")
        if not getattr(self, "use_prior_relative", True):
            prior_relative_mode = "none"
        return get_result_numeric_columns(
            getattr(self, "use_market_aggregates", True),
            getattr(self, "use_search_relative", True),
            getattr(self, "use_prior_relative", True),
            prior_relative_mode,
        )


def preprocess_expedia_data(
    df: pd.DataFrame,
    preprocessor: ExpediaPreprocessor | None = None,
    fit: bool = True,
    return_preprocessor: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, ExpediaPreprocessor]:
    """Create required flags, zero-impute, encode IDs, and normalize.

    By default this fits a new preprocessor on ``df``. For validation/test,
    pass the train-fitted ``preprocessor`` and ``fit=False``.
    """

    proc = preprocessor or ExpediaPreprocessor()
    processed = proc.fit_transform(df) if fit else proc.transform(df)
    if return_preprocessor:
        return processed, proc
    return processed
