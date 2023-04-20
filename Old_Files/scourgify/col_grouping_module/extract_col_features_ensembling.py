import pickle

import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np

from scourgify.col_grouping_module.Char_Distributions_Features import CharDistribution
from scourgify.col_grouping_module.Data_Type_Features import DataTypeFeatures
from scourgify.col_grouping_module.Embedding_Features import EmbeddingFeatures
from scourgify.col_grouping_module.NER_Parallel import NERFeatures
from scourgify.col_grouping_module.NER_huggingface import NER_huggingface
from scourgify.col_grouping_module.Single_Data_Type_Feature import SingleDataTypeFeatures
from scourgify.col_grouping_module.Value_Length_Features import ValueLengthStats

feature_methods = {'data_type_features': "SingleDataTypeFeatures", 'value_length_stats': "ValueLengthStats",
                   'char_distribution': "CharDistribution", 'NER': "NERFeatures"}


def extract_col_features(cols, headers, char_set):

    # Generate multiple clustering results using different feature methods
    cluster_labels = []
    for method in feature_methods.keys():
        print(method)
        # Create a new pipeline that includes only the current feature generation method and the normalizer
        pipeline_feature = Pipeline([
            ('feature_generator', FeatureUnion([
                (method, eval(feature_methods[method])(headers, char_set)),
            ])),
            ('normalizer', MinMaxScaler())
        ])

        # Generate features using the current pipeline
        X_features = pipeline_feature.fit_transform(cols)

        # Perform clustering using KMeans algorithm
        # kmeans = MiniBatchKMeans(n_clusters=7, random_state=0, reassignment_ratio=0, batch_size=256 * 64).fit(X_features)

        # Append the cluster labels to the list of cluster labels
        cluster_labels.append(kmeans.labels_)

    # Compute the consensus matrix
    n = len(X_features)
    consensus_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            agree_count = 0
            for labels in cluster_labels:
                if labels[i] == labels[j]:
                    agree_count += 1
            consensus_matrix[i][j] = agree_count / len(cluster_labels)
            consensus_matrix[j][i] = consensus_matrix[i][j]

    # Apply consensus clustering algorithm
    threshold = 0.8
    consensus_indices = np.where(np.sum(consensus_matrix > threshold, axis=1) >= 2)[0]
    consensus_labels = MiniBatchKMeans(n_clusters=7, random_state=0, reassignment_ratio=0, batch_size=256 * 64)\
        .fit_predict(X_features[consensus_indices])

    # Print the final cluster labels
    final_labels = np.zeros(n, dtype=np.int)
    final_labels[consensus_indices] = consensus_labels
    cols_per_cluster = {}
    l = set(final_labels)
    for i in l:
        if i in final_labels:
            cols_per_cluster[i] = []
            for col_idx in range(len(cols)):
                if final_labels[col_idx] == i:
                    cols_per_cluster[i].append(cols[col_idx])
    pickle.dump(cols_per_cluster, open(
        "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/scourgify/mediate_files/cols_per_cluster_sel.pkl",
        "wb"))
