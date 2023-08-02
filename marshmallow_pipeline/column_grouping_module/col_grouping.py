import logging
import os
import pickle

from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
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


def col_grouping(
    table_group, cols, max_n_col_groups, mediate_files_path, cg_enabled, col_grouping_alg, n_cores
):
    """
    Extracts features from a column
    Args:
        col: A column from a dataframe
        char_set: A set of characters that appear in the table group
        max_n_col_groups: The maximum number of column groups for each table group
        mediate_files_path: The path to the mediate files
        cg_enabled: A boolean that indicates whether column grouping is enabled
        col_grouping_alg: The column grouping algorithm (km for minibatch kmeans or hac for hierarchical agglomerative clustering - default: hac)
        n_cores: The number of cores to use for parallelization


    Returns:
        A dataframe of features

    """

    if cg_enabled:
        logging.info("Column grouping is enabled")
        logging.info("Grouping Columns in Table Group: %s", table_group)

        pipeline = Pipeline(
            [
                (
                    "feature_generator",
                    FeatureUnion(
                        [
                            (
                                "data_type_features",
                                DataTypeFeatures(),
                            ),
                            (
                                "value_length_stats",
                                ValueLengthStats(),
                            ),
                            (
                                "char_distribution",
                                CharTypeDistribution(),
                            ),
                        ]
                    ),
                ),
                ("normalizer", MinMaxScaler()),
            ]
        )

        X = pipeline.fit_transform(cols["col_value"])

        if max_n_col_groups > 1:
            if col_grouping_alg == "km":
                # For faster computations, you can set the batch_size greater than 256 * number of cores to enable parallelism on all cores (scikit-learn documentation)
                clusters = MiniBatchKMeans(
                    n_clusters=min(max_n_col_groups, len(X)),
                    batch_size=256 * n_cores,
                ).fit_predict(X)

            else:
                clusters = AgglomerativeClustering(n_clusters=min(max_n_col_groups, len(X))
                                               ).fit_predict(X)
        else:
            logging.info("The maximum number of column groups is less than 2")
            clusters = [0] * len(cols["col_value"])
            
    else:
        logging.info("Column grouping is disabled")
        clusters = [0] * len(cols["col_value"])

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
