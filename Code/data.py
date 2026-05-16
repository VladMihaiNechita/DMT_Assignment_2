"""Reading the parquet files and making transformer datasets."""

from __future__ import annotations

import gc

import pandas as pd

from .config import FORBIDDEN_TEST_COLUMNS, TEST_PATH, TRAIN_PATH
from .dataset import ExpediaSearchDataset
from .features import FeatureSpec
from .preprocessing import ExpediaPreprocessor


def fit_train_data() -> tuple[ExpediaSearchDataset, FeatureSpec, ExpediaPreprocessor]:
    print("Reading train data")
    train = pd.read_parquet(TRAIN_PATH).sort_values("srch_id", kind="mergesort").reset_index(drop=True)

    print("Making train features")
    # Market/prior/relative numeric features helped most. Bucket categorical
    # features were removed to keep the final feature set smaller.
    preprocessor = ExpediaPreprocessor(use_market_aggregates=True, use_bucket_categories=False)
    train = preprocessor.fit_transform(train)

    feature_spec = preprocessor.feature_spec
    if feature_spec is None:
        raise RuntimeError("The preprocessor did not create a feature spec.")

    train_dataset = ExpediaSearchDataset(train, feature_spec, has_labels=True, sort_by_search=False)
    del train
    gc.collect()

    print(
        "Feature counts:",
        feature_spec.num_search_numeric,
        "search numeric,",
        feature_spec.num_result_numeric,
        "result numeric,",
        feature_spec.num_categorical,
        "categorical",
    )
    return train_dataset, feature_spec, preprocessor


def make_test_data(preprocessor: ExpediaPreprocessor, feature_spec: FeatureSpec) -> ExpediaSearchDataset:
    print("\nReading test data")
    test = pd.read_parquet(TEST_PATH).sort_values("srch_id", kind="mergesort").reset_index(drop=True)

    bad_columns = sorted(FORBIDDEN_TEST_COLUMNS & set(test.columns))
    if bad_columns:
        raise ValueError(f"This must be the plain test file, but found columns: {bad_columns}")

    print("Making test features")
    # The final test data is only transformed with statistics learned from the
    # training data above.
    test = preprocessor.transform(test)
    test_dataset = ExpediaSearchDataset(test, feature_spec, has_labels=False, sort_by_search=False)
    del test
    gc.collect()
    return test_dataset
