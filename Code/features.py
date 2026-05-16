"""Feature definitions for the Expedia packed-transformer ranker.

The lists in this module intentionally mirror the assignment requirements.
Categorical ID columns are embedded by the model. The remaining columns are
fed as numeric features after flag generation, zero imputation, and selected
standardization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


SEARCH_CATEGORICAL_COLUMNS = [
    "site_id",
    "visitor_location_country_id",
]

RESULT_CATEGORICAL_COLUMNS = [
    "prop_country_id",
    "prop_id",
    "srch_destination_id",
]

ORDINAL_CATEGORICAL_COLUMNS = [
    "prop_starrating_ord",
    "prop_review_score_ord",
] + [
    f"comp{i}_{suffix}_ord"
    for i in range(1, 9)
    for suffix in ("rate", "inv")
]

BUCKET_CATEGORICAL_COLUMNS = [
    "price_bucket_ord",
    "booking_window_bucket_ord",
    "stay_length_bucket_ord",
    "price_rank_bucket_ord",
    "price_diff_rank_bucket_ord",
    "star_rank_bucket_ord",
    "loc2_rank_bucket_ord",
]
MARKET_V4_CATEGORICAL_COLUMNS = (
    SEARCH_CATEGORICAL_COLUMNS
    + RESULT_CATEGORICAL_COLUMNS
    + ORDINAL_CATEGORICAL_COLUMNS
)

SEARCH_BASE_COLUMNS = [
    "month_of_year",
    "day_of_week",
    "week_of_year",
    "hour_of_day",
    "hour_bucket",
    "time_epoch",
    "random_bool",
    "n_props",
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
]

SEARCH_FLAG_COLUMNS = [
    "is_NULL_visitor_hist_starrating",
    "is_NULL_visitor_hist_adr_usd",
]

RESULT_BASE_COLUMNS = [
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "promotion_flag",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "srch_query_affinity_score",
    "orig_destination_distance",
]

COMPETITOR_COLUMNS = [
    f"comp{i}_{suffix}"
    for i in range(1, 9)
    for suffix in ("rate", "inv", "rate_percent_diff")
]

RESULT_FLAG_COLUMNS = [
    "is_0_prop_starrating",
    "is_NULL_prop_review_score",
    "is_0_prop_review_score",
    "is_NULL_prop_location_score2",
    "is_0_prop_log_historical_price",
    "is_NULL_srch_query_affinity_score",
    "is_NULL_orig_destination_distance",
]

COMPETITOR_FLAG_COLUMNS = [
    f"is_NULL_comp{i}_{suffix}"
    for i in range(1, 9)
    for suffix in ("rate", "inv", "rate_percent_diff")
]

PRICE_FEATURE_COLUMNS = [
    "log_price",
    "price_per_night",
    "price_per_room",
    "price_per_adult",
    "price_per_person",
    "price_vs_historical",
    "ump",
    "price_ratio_historical",
    "total_search_size",
]

MATCH_FEATURE_COLUMNS = [
    "star_gap",
    "adr_gap",
    "domestic",
]

COMPETITOR_AGG_COLUMNS = [
    "comp_rate_mean",
    "comp_n_competitors",
    "comp_n_better",
    "comp_n_worse",
    "comp_inv_mean",
    "comp_pct_mean",
    "comp_pct_max",
    "comp_pct_min",
]

COMPOSITE_FEATURE_COLUMNS = [
    "star_x_review",
    "loc_score1_x_score2",
]

MARKET_AGG_FEATURE_COLUMNS = [
    "prop_price_mean",
    "prop_price_std",
    "prop_price_count_log",
    "prop_log_price_mean",
    "prop_promotion_rate",
    "price_vs_prop_mean",
    "log_price_vs_prop_mean",
    "promotion_vs_prop_rate",
    "prop_dest_price_mean",
    "prop_dest_price_std",
    "prop_dest_count_log",
    "prop_dest_log_price_mean",
    "price_vs_prop_dest_mean",
    "log_price_vs_prop_dest_mean",
    "dest_price_mean",
    "dest_price_std",
    "dest_count_log",
    "dest_log_price_mean",
    "dest_promotion_rate",
    "dest_star_mean",
    "dest_review_mean",
    "dest_loc_score1_mean",
    "price_vs_dest_mean",
    "log_price_vs_dest_mean",
    "promotion_vs_dest_rate",
    "star_vs_dest_mean",
    "review_vs_dest_mean",
    "loc_score1_vs_dest_mean",
]

SEARCH_RELATIVE_BASE_COLUMNS = [
    "price_usd",
    "log_price",
    "price_per_night",
    "price_per_person",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "promotion_flag",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "price_vs_historical",
    "ump",
    "star_x_review",
    "loc_score1_x_score2",
]

SEARCH_RELATIVE_SUFFIXES = [
    "diff_mean",
    "diff_min",
    "zscore",
    "range_pos",
    "rank_pct",
]

SEARCH_RELATIVE_COLUMNS = [
    f"{col}_{suffix}"
    for col in SEARCH_RELATIVE_BASE_COLUMNS
    for suffix in SEARCH_RELATIVE_SUFFIXES
]

POSITION_AGG_COLUMNS = [
    "prop_mean_position",
    "prop_median_position",
    "prop_position_count",
]

HISTORICAL_PRIOR_KEYS = [
    ("prop_id",),
    ("prop_id", "srch_destination_id"),
    ("prop_id", "srch_length_of_stay"),
    ("prop_id", "srch_booking_window"),
    ("prop_id", "srch_room_count"),
    ("prop_id", "month_of_year"),
    ("prop_id", "hour_bucket"),
    ("prop_id", "site_id"),
]

HISTORICAL_PRIOR_COLUMNS = [
    f"{'_x_'.join(keys)}_{suffix}"
    for keys in HISTORICAL_PRIOR_KEYS
    for suffix in ("booking_rate", "click_rate", "relevance_mean", "impressions_log")
]

NONRANDOM_PRIOR_COLUMNS = [
    "prop_id_booking_rate_nonrandom",
    "prop_id_click_rate_nonrandom",
    "prop_id_relevance_mean_nonrandom",
    "prop_id_impressions_log_nonrandom",
]

PRIOR_COLUMNS = HISTORICAL_PRIOR_COLUMNS + NONRANDOM_PRIOR_COLUMNS
PRIOR_RELATIVE_COLUMNS = [
    f"{col}_{suffix}"
    for col in PRIOR_COLUMNS
    for suffix in ("diff_mean", "rank_pct")
]
CORE_PRIOR_RELATIVE_COLUMNS = [
    f"{col}_{suffix}"
    for col in (
        [
            f"{'_x_'.join(keys)}_{suffix}"
            for keys in (("prop_id",), ("prop_id", "srch_destination_id"))
            for suffix in ("booking_rate", "click_rate", "relevance_mean", "impressions_log")
        ]
        + NONRANDOM_PRIOR_COLUMNS
    )
    for suffix in ("diff_mean", "rank_pct")
]
CONTEXT_RANK_PRIOR_RELATIVE_COLUMNS = [
    f"{'_x_'.join(keys)}_{rate_suffix}_rank_pct"
    for keys in HISTORICAL_PRIOR_KEYS
    for rate_suffix in ("booking_rate", "click_rate", "relevance_mean")
]
PRIOR_RELATIVE_MODE_COLUMNS = {
    "all": PRIOR_RELATIVE_COLUMNS,
    "core": CORE_PRIOR_RELATIVE_COLUMNS,
    "context_rank": CONTEXT_RANK_PRIOR_RELATIVE_COLUMNS,
    "none": [],
}


def get_prior_relative_columns(prior_relative_mode: str) -> list[str]:
    try:
        return list(PRIOR_RELATIVE_MODE_COLUMNS[prior_relative_mode])
    except KeyError as exc:
        raise ValueError(f"Unknown prior_relative_mode={prior_relative_mode!r}") from exc

SEARCH_NUMERIC_COLUMNS = SEARCH_BASE_COLUMNS + SEARCH_FLAG_COLUMNS
RESULT_NUMERIC_COLUMNS = (
    RESULT_BASE_COLUMNS
    + COMPETITOR_COLUMNS
    + PRICE_FEATURE_COLUMNS
    + MATCH_FEATURE_COLUMNS
    + COMPETITOR_AGG_COLUMNS
    + COMPOSITE_FEATURE_COLUMNS
    + MARKET_AGG_FEATURE_COLUMNS
    + SEARCH_RELATIVE_COLUMNS
    + POSITION_AGG_COLUMNS
    + PRIOR_COLUMNS
    + PRIOR_RELATIVE_COLUMNS
    + RESULT_FLAG_COLUMNS
    + COMPETITOR_FLAG_COLUMNS
)
CATEGORICAL_COLUMNS = (
    MARKET_V4_CATEGORICAL_COLUMNS
    + BUCKET_CATEGORICAL_COLUMNS
)


def get_categorical_columns(use_bucket_categories: bool = True) -> list[str]:
    columns = list(MARKET_V4_CATEGORICAL_COLUMNS)
    if use_bucket_categories:
        columns.extend(BUCKET_CATEGORICAL_COLUMNS)
    return columns


def get_result_numeric_columns(
    use_market_aggregates: bool = True,
    use_search_relative: bool = True,
    use_prior_relative: bool = True,
    prior_relative_mode: str | None = None,
) -> list[str]:
    if prior_relative_mode is None:
        prior_relative_mode = "all" if use_prior_relative else "none"
    if not use_prior_relative:
        prior_relative_mode = "none"

    columns = (
        RESULT_BASE_COLUMNS
        + COMPETITOR_COLUMNS
        + PRICE_FEATURE_COLUMNS
        + MATCH_FEATURE_COLUMNS
        + COMPETITOR_AGG_COLUMNS
        + COMPOSITE_FEATURE_COLUMNS
    )
    if use_market_aggregates:
        columns += MARKET_AGG_FEATURE_COLUMNS
    if use_search_relative:
        columns += SEARCH_RELATIVE_COLUMNS
    columns += POSITION_AGG_COLUMNS + PRIOR_COLUMNS
    columns += get_prior_relative_columns(prior_relative_mode)
    columns += RESULT_FLAG_COLUMNS + COMPETITOR_FLAG_COLUMNS
    return list(columns)

NORMALIZE_COLUMNS = [
    "month_of_year",
    "day_of_week",
    "week_of_year",
    "hour_of_day",
    "hour_bucket",
    "time_epoch",
    "n_props",
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_query_affinity_score",
    "orig_destination_distance",
] + [f"comp{i}_rate_percent_diff" for i in range(1, 9)] + [
    col
    for col in (
        PRICE_FEATURE_COLUMNS
        + MATCH_FEATURE_COLUMNS
        + COMPETITOR_AGG_COLUMNS
        + COMPOSITE_FEATURE_COLUMNS
        + MARKET_AGG_FEATURE_COLUMNS
        + SEARCH_RELATIVE_COLUMNS
        + POSITION_AGG_COLUMNS
        + PRIOR_COLUMNS
        + PRIOR_RELATIVE_COLUMNS
    )
    if col not in {"domestic"}
]


@dataclass(frozen=True)
class FeatureSpec:
    """All model-facing feature columns and categorical cardinalities."""

    search_numeric: list[str]
    result_numeric: list[str]
    categorical: list[str]
    categorical_cardinalities: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def num_search_numeric(self) -> int:
        return len(self.search_numeric)

    @property
    def num_result_numeric(self) -> int:
        return len(self.result_numeric)

    @property
    def num_categorical(self) -> int:
        return len(self.categorical)
