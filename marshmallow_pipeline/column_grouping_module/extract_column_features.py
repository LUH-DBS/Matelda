import logging
import os
import pickle

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
from sklearn.metrics import euclidean_distances
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

    # Calculate the Euclidean distance matrix
    distance_matrix = euclidean_distances(X)
    similarity_matrix = 1 / (1 + distance_matrix)
    # Calculate median similarity value
    median_similarity = np.median(similarity_matrix)
    # Prune edges below median similarity
    similarity_matrix = np.where(
        similarity_matrix > median_similarity, similarity_matrix, 0
    )
    # Create a graph from the distance matrix
    graph = nx.Graph(similarity_matrix)

    # Set the range of resolution parameter values to sweep
    resolution_range = np.arange(1, 2.1, 0.1)  # adjust the range as desired

    best_communities = None
    for resolution in resolution_range:
        communities = nx_comm.louvain_communities(graph, resolution=resolution)
        if len(communities) <= max_n_col_groups:
            best_communities = communities
        else:
            logging.info(
                "resolution %s, Number of communities is greater than the maximum number of column groups",
                resolution,
            )

    if best_communities is None:
        logging.info(
            "Number of communities is greater than the maximum number of column groups"
        )
        return None

    logging.info("**********Table Group*********: %s", table_group)
    logging.info("Communities: %s", best_communities)
    logging.info("Number of communities: %s", len(best_communities))

    # Convert the communities to a dictionary format
    comm_dict = {}
    for i, comm in enumerate(best_communities):
        for node in comm:
            comm_dict[node] = i

    cols_per_cluster = {}
    col_group_df = {
        "column_cluster_label": [],
        "col_value": [],
        "table_id": [],
        "table_path": [],
        "table_cluster": [],
        "col_id": [],
    }
    community_labels = set(range(len(best_communities)))
    for i in community_labels:
        comm = best_communities[i]
        cols_per_cluster[i] = []
        for c in best_communities[i]:
            cols_per_cluster[i].append(cols["col_value"][c])
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
