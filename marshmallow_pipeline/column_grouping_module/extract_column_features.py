import os
import pickle

from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler

from marshmallow_pipeline.column_grouping_module.chartypes_distributions_features import (
    CharTypeDistribution,
)
from marshmallow_pipeline.column_grouping_module.data_type_features import (
    DataTypeFeatures,
)
from marshmallow_pipeline.column_grouping_module.value_length_features import (
    ValueLengthStats,
)


def extract_column_features(
    table_group, cols, char_set, max_n_col_groups, mediate_files_path
):
    """
    Extracts features from a column
    Args:
        col: A column from a dataframe

    Returns:
        A dataframe of features

    """
    # Feature weighting
    w = {"data_type_features": 0.7, "value_length_stats": 0.1, "char_distribution": 0.2}

    pipeline = Pipeline(
        [
            (
                "feature_generator",
                FeatureUnion(
                    [
                        (
                            "data_type_features",
                            DataTypeFeatures(w["data_type_features"]),
                        ),
                        (
                            "value_length_stats",
                            ValueLengthStats(w["value_length_stats"]),
                        ),
                        (
                            "char_distribution",
                            CharTypeDistribution(char_set, w["char_distribution"]),
                        ),
                    ]
                ),
            ),
            ("normalizer", MinMaxScaler()),
        ]
    )

    X = pipeline.fit_transform(cols["col_value"])

    clusters = MiniBatchKMeans(
        n_clusters=min(max_n_col_groups, len(X)),
        random_state=0,
        reassignment_ratio=0,
        batch_size=256 * 64,
    ).fit_predict(X)

    cols_per_cluster = {}
    cols_per_cluster_values = {}
    for col, col_clu in enumerate(clusters):
        if col_clu not in cols_per_cluster:
            cols_per_cluster[col_clu] = []
        cols_per_cluster[col_clu].append(col)
        if col_clu not in cols_per_cluster_values:
            cols_per_cluster_values[col_clu] = []
        cols_per_cluster_values[col_clu].append(cols["col_value"][col])

    col_group_df = {
        "column_cluster_label": [],
        "col_value": [],
        "table_id": [],
        "table_path": [],
        "table_cluster": [],
        "col_id": [],
    }
    for i in set(clusters):
        for c in cols_per_cluster[i]:
            col_group_df["column_cluster_label"].append(i)
            col_group_df["col_value"].append(cols["col_value"][c])
            col_group_df["table_id"].append(cols["table_id"][c])
            col_group_df["table_path"].append(cols["table_path"][c])
            col_group_df["table_cluster"].append(table_group)
            col_group_df["col_id"].append(cols["col_id"][c])

    col_grouping_res = os.path.join(mediate_files_path, "col_grouping_res")
    cols_per_clu = os.path.join(col_grouping_res, "cols_per_clu")
    col_df_res = os.path.join(col_grouping_res, "col_df_res")

    os.makedirs(col_grouping_res, exist_ok=True)
    os.makedirs(cols_per_clu, exist_ok=True)
    os.makedirs(col_df_res, exist_ok=True)

    with open(
        os.path.join(cols_per_clu, f"cols_per_cluster_{table_group}.pkl"), "wb+"
    ) as file:
        pickle.dump(cols_per_cluster, file)

    with open(
        os.path.join(col_df_res, f"col_df_labels_cluster_{table_group}.pickle"), "wb+"
    ) as file:
        pickle.dump(col_group_df, file)

    return col_group_df
