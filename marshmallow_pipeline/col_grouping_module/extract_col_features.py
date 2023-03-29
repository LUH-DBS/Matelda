import pickle

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from col_grouping_module.CharTypes_Distributions_Features import CharTypeDistribution
from col_grouping_module.Char_Distributions_Features import CharDistribution
from col_grouping_module.Single_Data_Type_Feature import SingleDataTypeFeatures
from col_grouping_module.Value_Length_Features import ValueLengthStats
import networkx.algorithms.community as nx_comm


def extract_col_features(table_group, cols, char_set):
    """
    Extracts features from a column
    Args:
        col: A column from a dataframe

    Returns:
        A dataframe of features

    """
    pipeline = Pipeline([
        ('feature_generator', FeatureUnion([
            ('data_type_features', SingleDataTypeFeatures()),
            ('value_length_stats', ValueLengthStats()),
            ('char_distribution', CharDistribution(char_set)),
        ])),
        ('normalizer', MinMaxScaler())
    ])

    X = pipeline.fit_transform(cols["col_value"])
    # cl = DBSCAN(min_samples=2).fit(X)
    # labels = cl.labels_
    # # cl = MiniBatchKMeans(n_clusters=7, random_state=0, reassignment_ratio=0, batch_size=256 * 64).fit(X)
    # # labels = cl.labels_

    # Calculate the Euclidean distance matrix
    # distance_matrix = euclidean_distances(X)
    # similarity_matrix = 1 / (1 + distance_matrix)
    similarity_matrix = cosine_similarity(X)
    # Calculate median similarity value
    median_similarity = np.median(similarity_matrix)
    # Prune edges below median similarity
    similarity_matrix = np.where(similarity_matrix > median_similarity, similarity_matrix, 0)
    # Create a graph from the distance matrix
    graph = nx.Graph(similarity_matrix)
    communities = nx_comm.louvain_communities(graph, resolution=1.5)
    print("**********Table Group*********:", table_group)
    print("Communities:", communities)

    cols_per_cluster = {}
    col_group_df = {"column_cluster_label": [], "col_value": [], "table_id": [], "table_path": [], "table_cluster": [],
                    "col_id": []}
    l = set(range(len(communities)))
    for i in l:
        cols_per_cluster[i] = []
        for c in communities[i]:
            cols_per_cluster[i].append(cols["col_value"][c])
            col_group_df["column_cluster_label"].append(i)
            col_group_df["col_value"].append(cols["col_value"][c])
            col_group_df["table_id"].append(cols["table_id"][c])
            col_group_df["table_path"].append(cols["table_path"][c])
            col_group_df["table_cluster"].append(table_group)
            col_group_df["col_id"].append(cols["col_id"][c])

    pickle.dump(cols_per_cluster,
                open("/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/mediate_files/col_grouping_res/cols_per_cluster_{}.pkl".format(table_group), "wb"))
    pickle.dump(col_group_df,
                open("/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/mediate_files/col_grouping_res/col_df_labels_cluster_{}.pickle".format(table_group), "wb"))
    return col_group_df
