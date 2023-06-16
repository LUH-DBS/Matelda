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
from sklearn.mixture import GaussianMixture

logger = logging.getLogger()

def extract_col_features(table_group, cols, char_set, max_n_col_groups, mediate_files_path):
    """
    Extracts features from a column
    Args:
        col: A column from a dataframe

    Returns:
        A dataframe of features

    """    
    pipeline = Pipeline([
        ('feature_generator', FeatureUnion([
            ('data_type_features', DataTypeFeatures()),
            ('value_length_stats', ValueLengthStats()),
            ('char_distribution', CharTypeDistribution(char_set)),
        ])),
        ('normalizer', MinMaxScaler())
    ])

    X = pipeline.fit_transform(cols["col_value"])

    clusters = MiniBatchKMeans(n_clusters= min(max_n_col_groups, len(X)), random_state=0, reassignment_ratio=0, batch_size = 256 * 64).fit_predict(X)

    cols_per_cluster = {}
    cols_per_cluster_values = {}
    for col, col_clu in enumerate(clusters):
        if col_clu not in cols_per_cluster:
            cols_per_cluster[col_clu] = []
        cols_per_cluster[col_clu].append(col)
        if col_clu not in cols_per_cluster_values:
            cols_per_cluster_values[col_clu] = []
        cols_per_cluster_values[col_clu].append(cols["col_value"][col])

    col_group_df = {"column_cluster_label": [], "col_value": [], "table_id": [], "table_path": [], "table_cluster": [], "col_id": []}
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
    if not os.path.exists(col_grouping_res):
        os.makedirs(col_grouping_res)
    if not os.path.exists(cols_per_clu):
        os.makedirs(cols_per_clu)
    if not os.path.exists(col_df_res):
        os.makedirs(col_df_res)
    pickle.dump(cols_per_cluster_values,
                open(os.path.join(cols_per_clu, "cols_per_cluster_{}.pkl".format(table_group)), "wb"))
    pickle.dump(col_group_df,
                open(os.path.join(col_df_res, "col_df_labels_cluster_{}.pickle".format(table_group)), "wb"))
    return col_group_df
