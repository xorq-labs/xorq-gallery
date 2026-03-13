"""Load a fitted pipeline from the xorq catalog, remap its column references
to FAA feature names, then fit and predict on FAA flight data.

Source pipeline (Titanic):
  num slot: ["age", "fare"]            → FAA: ["distance", "flight_time", "dep_hour"]
  cat slot: ["embarked", "sex", "pclass"] → FAA: ["carrier", "origin", "destination"]
"""

import toolz
import xorq.api as xo
from git import Repo
from sklearn.metrics import accuracy_score, f1_score
from xorq.catalog.catalog import Catalog
from xorq.expr.ml.sklearn_utils import ColumnRemapper

from xorq_gallery.sklearn.utils import load_build_paths_json_cache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://pub-a45a6a332b4646f2a6f44775695c64df.r2.dev"

NUMERIC_FEATURES = ("distance", "flight_time", "dep_hour")
CATEGORICAL_FEATURES = ("carrier", "origin", "destination")
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "is_late"
PRED_COL = "pred"
UNIQUE_KEY = "id2"

TEST_SIZE = 0.2
RANDOM_STATE = 42
LATE_THRESHOLD = 15
TOP_N_AIRPORTS = 5

CATALOG_NAME = "xorq-gallery-sklearn"
SCRIPT_NAME, EXPR_NAME = "plot_column_transformer_mixed_types.py", "xorq_ct_lr_preds"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data():
    con = xo.connect()
    flights = xo.deferred_read_parquet(
        f"{BASE_URL}/flights.parquet", con, table_name="flights"
    )

    expr = (
        flights.select(
            "id2",
            "carrier",
            "origin",
            "destination",
            "distance",
            "flight_time",
            "dep_time",
            "arr_delay",
            "cancelled",
        )
        .filter(lambda t: t.cancelled == "N")
        .drop("cancelled")
        .filter(lambda t: t.arr_delay.notnull() & t.dep_time.notnull())
        .mutate(
            dep_hour=lambda t: t.dep_time.hour().cast("float64"),
            is_late=lambda t: (t.arr_delay > LATE_THRESHOLD).cast("int64"),
        )
    )

    top_airports = (
        expr.group_by("origin")
        .agg(cnt=lambda t: t.count())
        .order_by(xo.desc("cnt"))
        .limit(TOP_N_AIRPORTS)
        .origin.execute()
        .tolist()
    )
    return expr.mutate(
        origin=lambda t: xo.ifelse(t.origin.isin(top_airports), t.origin, "OTHER"),
        destination=lambda t: xo.ifelse(
            t.destination.isin(top_airports), t.destination, "OTHER"
        ),
    ).select(UNIQUE_KEY, *FEATURE_COLS, TARGET_COL)


def split_data(data):
    return xo.train_test_splits(
        data,
        test_sizes=TEST_SIZE,
        unique_key=UNIQUE_KEY,
        random_seed=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_remapped_pipeline():
    build_name = toolz.get_in(
        (SCRIPT_NAME, EXPR_NAME),
        load_build_paths_json_cache(),
    ).split("/")[-1]
    catalog = Catalog(Repo(f"./.xorq/git-catalogs/{CATALOG_NAME}"))
    catalog_entry = catalog.get_catalog_entry(build_name)
    print(
        f"\nLoading pipeline from catalog '{CATALOG_NAME}/{build_name}' and remapping columns..."
    )
    return catalog_entry.expr.ls.pipeline.remap_columns(
        {
            "preprocessor/num": list(NUMERIC_FEATURES),
            "preprocessor/cat": list(CATEGORICAL_FEATURES),
        }
    ).remap_params({"classifier__class_weight": "balanced"})


# ---------------------------------------------------------------------------
# Module-level deferred exprs
# ---------------------------------------------------------------------------

data = load_data()
train_table, test_table = split_data(data)

pipeline = load_remapped_pipeline()
fitted = pipeline.fit(train_table, features=FEATURE_COLS, target=TARGET_COL)
preds_expr = fitted.predict(test_table, name=PRED_COL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading FAA flights data...")
    df = data.execute()
    print(f"  {len(df):,} flights  |  late-arrival rate: {df[TARGET_COL].mean():.1%}")

    print(
        f"  column refs after remap: {ColumnRemapper.list_column_refs(pipeline.instance)}"
    )

    print("\nExecuting deferred predictions...")
    preds_df = preds_expr.execute()

    y_test = preds_df[TARGET_COL].values
    y_pred = preds_df[PRED_COL].values

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n  accuracy={acc:.4f}  F1={f1:.4f}")


if __name__ == "__main__":
    main()
