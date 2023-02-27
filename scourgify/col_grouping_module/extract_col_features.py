import pickle

import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold

from scourgify.col_grouping_module.Char_Distributions_Features import CharDistribution
from scourgify.col_grouping_module.Data_Type_Features import DataTypeFeatures
from scourgify.col_grouping_module.Embedding_Features import EmbeddingFeatures
from scourgify.col_grouping_module.Value_Length_Features import ValueLengthStats


def extract_col_features(cols, char_set):
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
            ('char_distribution', CharDistribution(char_set)),
            ('embedding', EmbeddingFeatures())
        ])),
        ('normalizer', MinMaxScaler()),
        ('feature_selector', VarianceThreshold(threshold=0.0))
    ])

    X = pipeline.fit_transform(cols)
    # cl = DBSCAN(min_samples=2).fit(X)
    # dbscan.labels_
    cl = MiniBatchKMeans(n_clusters=7, random_state=0, reassignment_ratio=0, batch_size=256 * 64).fit(X)
    labels = cl.labels_
    cols_per_cluster = {}
    l = set(labels)
    for i in l:
        if i in labels:
            cols_per_cluster[i] = []
            for col_idx in range(len(cols)):
                if labels[col_idx] == i:
                    cols_per_cluster[i].append(cols[col_idx])
    pickle.dump(cols_per_cluster, open("/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/scourgify/mediate_files/cols_per_cluster_sel.pkl", "wb"))
    return
