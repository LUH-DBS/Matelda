import pickle

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scourgify.col_grouping_module.CharTypes_Distributions_Features import CharTypeDistribution
from scourgify.col_grouping_module.Char_Distributions_Features import CharDistribution
from scourgify.col_grouping_module.NER_Parallel import NERFeatures
from scourgify.col_grouping_module.Single_Data_Type_Feature import SingleDataTypeFeatures
from scourgify.col_grouping_module.Value_Length_Features import ValueLengthStats
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

    X = pipeline.fit_transform(cols)
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
    communities = nx_comm.louvain_communities(graph)
    print("**********Table Group*********:", table_group)
    print("Communities:", communities)

    cols_per_cluster = {}
    l = set(range(len(communities)))
    for i in l:
        cols_per_cluster[i] = []
        for c in communities[i]:
            cols_per_cluster[i].append(cols[c])
    pickle.dump(cols_per_cluster,
                open("/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/"
                     "scourgify/mediate_files/cols_per_cluster_{}.pkl".format(table_group), "wb"))
    return
