import logging
import os
import pickle

import networkx as nx
import community
import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from col_grouping_module.CharTypes_Distributions_Features import CharTypeDistribution
from col_grouping_module.Char_Distributions_Features import CharDistribution
from col_grouping_module.Single_Data_Type_Feature import SingleDataTypeFeatures
from col_grouping_module.Data_Type_Features import DataTypeFeatures
from col_grouping_module.Value_Length_Features import ValueLengthStats
import networkx.algorithms.community as nx_comm

logger = logging.getLogger()

def extract_col_features(table_group, cols, char_set, max_n_col_groups, mediate_files_path):
    """
    Extracts features from a column
    Args:
        col: A column from a dataframe

    Returns:
        A dataframe of features

    """
    # Feature weighting
    w = {'data_type_features': 0.7, 'value_length_stats': 0.1, 'char_distribution': 0.2}
    
    pipeline = Pipeline([
        ('feature_generator', FeatureUnion([
            ('data_type_features', DataTypeFeatures(w["data_type_features"])),
            ('value_length_stats', ValueLengthStats(w["value_length_stats"])),
            ('char_distribution', CharTypeDistribution(char_set, w["char_distribution"])),
        ])),
        ('normalizer', MinMaxScaler())
    ])

    X = pipeline.fit_transform(cols["col_value"])

    
    # Calculate the Euclidean distance matrix
    distance_matrix = euclidean_distances(X)
    similarity_matrix = 1 / (1 + distance_matrix)
    # Calculate median similarity value
    median_similarity = np.median(similarity_matrix)
    # Prune edges below median similarity
    similarity_matrix = np.where(similarity_matrix > median_similarity, similarity_matrix, 0)
    # Create a graph from the distance matrix
    graph = nx.Graph(similarity_matrix)

    # Set the range of resolution parameter values to sweep
    resolution_range = np.arange(1, 2.1, 0.1) # adjust the range as desired

    best_communities = None
    for resolution in resolution_range:
        communities = nx_comm.louvain_communities(graph, resolution=resolution)
        if len(communities) <= max_n_col_groups:
            best_communities = communities
        else:
            logger.info("resolution {}, Number of communities is greater than the maximum number of column groups".format(resolution))

    if best_communities is None:
        logger.info("Number of communities is greater than the maximum number of column groups")
        return None
    
    logger.info("**********Table Group*********: {}".format(table_group))
    logger.info("Communities: {}".format(best_communities))
    logger.info("Number of communities: {}".format(len(best_communities)))

    # Convert the communities to a dictionary format
    comm_dict = {}
    for i, comm in enumerate(best_communities):
        for node in comm:
            comm_dict[node] = i

    cols_per_cluster = {}
    col_group_df = {"column_cluster_label": [], "col_value": [], "table_id": [], "table_path": [], "table_cluster": [], "col_id": []}
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
    if not os.path.exists(col_grouping_res):
        os.makedirs(col_grouping_res)
    if not os.path.exists(cols_per_clu):
        os.makedirs(cols_per_clu)
    if not os.path.exists(col_df_res):
        os.makedirs(col_df_res)
    pickle.dump(cols_per_cluster,
                open(os.path.join(cols_per_clu, "cols_per_cluster_{}.pkl".format(table_group)), "wb"))
    pickle.dump(col_group_df,
                open(os.path.join(col_df_res, "col_df_labels_cluster_{}.pickle".format(table_group)), "wb"))
    return col_group_df
