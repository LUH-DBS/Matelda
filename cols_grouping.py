import logging
import shutil
import os
import pickle
import csv
import numpy as np
import pandas as pd
from openclean.profiling.dataset import dataset_profile
from sklearn.cluster import DBSCAN

import nltk
from messytables import CSVTableSet, type_guess
import spacy
from sklearn.cluster import KMeans
from kmeans_interp.kmeans_feature_imp import KMeansInterp
from statistics import median

logger = logging.getLogger()

type_dicts = {'Integer': 0, 'Decimal': 1, 'String': 2, 'Date': 3, 'Bool': 4, 'Time': 5, 'Currency': 6, 'Percentage': 7}

def get_clusters_dict(df):
    clusters_dict = {}

    clusters = df["cluster"].unique()
    for i in clusters:
        clusters_dict[i] = []

    row_iterator = df.iterrows()
    for i, row in row_iterator:
        clusters_dict[row['cluster']].append((row['table_id'], row['parent'], row['table_name']))

    return clusters_dict


def get_col_df(sandbox_path, cluster, labels_dict_path):
    column_dict = {'table_id': [], 'col_id': [], 'col_value': [], 'col_gt': [], 'col_type': []}
    lake_labels_dict = extract_labels(labels_dict_path)

    for table in cluster:
        print(table)
        parent_path = os.path.join(sandbox_path, table[1])
        table_path = os.path.join(parent_path, table[2] + "/dirty.csv")
        df = pd.read_csv(table_path, quoting=csv.QUOTE_ALL)

        table_file = open(table_path, 'rb')
        table_set = CSVTableSet(table_file)
        row_set = table_set.tables[0]
        types = type_guess(row_set.sample)

        for column_idx, column in enumerate(df.columns.tolist()):
            column_dict['table_id'].append(table[0])
            column_dict['col_id'].append(column_idx)
            column_dict['col_value'].append(df[column].tolist())
            column_dict['col_gt'].append(lake_labels_dict[(table[0], column_idx)].values)
            column_dict['col_type'].append(types[column_idx])
            

    col_df = pd.DataFrame.from_dict(column_dict)

    return col_df


def get_col_features(col_df, ner_model_name):
    col_features = []
    characters_dictionary = {}
    tokens_dictionary = {}
    sum_value_length_dictionary = {}

    # feature_names = ['col_type', 'uniqueness', 'emptyValueCount']
    feature_names = ['col_type']

    for i in range(col_df.shape[0]):
        features = []
        profiles = dataset_profile(pd.DataFrame(col_df['col_value'][i]))
        # features.append(profiles.stats()['uniqueness'][0])
        # features.append(profiles[0]['stats']['emptyValueCount'])

        if col_df['col_type'][i]:
            col_type = str(col_df['col_type'][i])
            if 'Date' in col_type:
                features.append(type_dicts['Date'])
            elif 'Time' in col_type:
                features.append(type_dicts['Type'])
            else:
                features.append(type_dicts[col_type])
        else:
            features.append(-1)
        

        for j in range(len(features)):
            if features[j] is None:
                features[j] = -1
        col_features.append(features)

        value_length_sum = []
        for value in col_df['col_value'][i]:
            char_list = list(set(list(str(value))))
            if ' ' in char_list:
                char_list.remove(' ')
            for character in char_list:
                if character not in characters_dictionary:
                    characters_dictionary[character] = 0.0
                characters_dictionary[character] += 1.0
            value_length_sum.append(len(str(value)))
            tokens = str(value).split()
            if ' ' in tokens:
                tokens.remove(' ')
            for token in tokens:
                if token not in tokens_dictionary:
                    tokens_dictionary[token] = 0.0
                tokens_dictionary[token] += 1.0
    
        sum_value_length_dictionary[i] = value_length_sum

    for key in sum_value_length_dictionary.keys():
        # sum_value_length_dictionary[key] = sum_value_length_dictionary[key]/len(col_df['col_value'][key])
        sum_value_length_dictionary[key] = median(sum_value_length_dictionary[key])

    for token in list(tokens_dictionary.keys()):
        if token in characters_dictionary.keys():
            del tokens_dictionary[token]


    for i in range(col_df.shape[0]):
        column_profile = {
            "characters": {ch: characters_dictionary[ch] / len(col_df['col_value'][i]) for ch in characters_dictionary},
            "tokens": {v: tokens_dictionary[v] / len(col_df['col_value'][i]) for v in tokens_dictionary},
            "avg_value_length": sum_value_length_dictionary[i]
        }
        char_list = list(column_profile["characters"].values())
        tokens_list = list(column_profile["tokens"].values())

        for char in char_list:
            col_features[i].append(char)
            
        for token in tokens_list:
            col_features[i].append(token)

        col_features[i].append(column_profile['avg_value_length'])

    feature_names.extend([c for c in characters_dictionary.keys()])
    feature_names.extend([str(v) for v in tokens_dictionary.keys()])
    feature_names.append("avg_value_length")
    return col_features, feature_names


def cluster_cols_auto(col_features, auto_clustering_enabled, feature_names):
    # TODO: dbscan params config
    reduced_features = []
    # TODO
    for col_feature in col_features:
        reduced_features.append(col_feature[8:])
    # columns = ['table_id', 'col_id', 'totalValueCount', 'emptyValueCount', 'distinctValueCount',
    #            'uniqueness', 'entropy', 'data_type_code']
    vocabulary = [str(i) for i in range(8, len(col_features[0]))]
    columns = feature_names + vocabulary

    if auto_clustering_enabled:
        clustering_results = DBSCAN(eps=0.7, min_samples=10).fit(reduced_features)
        col_labels_df = pd.DataFrame(col_features, columns=columns)
        col_labels_df['column_cluster_label'] = pd.DataFrame(clustering_results.labels_)
    else:
        col_labels_df = pd.DataFrame(col_features, columns=columns)
        ones_ = np.ones(len(col_features))
        col_labels_df['column_cluster_label'] = pd.DataFrame(ones_)

    number_of_clusters = len(col_labels_df['column_cluster_label'].unique())

    return col_labels_df, number_of_clusters

def cluster_cols(col_features, auto_clustering_enabled, feature_names):
    
    # vocabulary = [str(i) for i in range(3, len(col_features[0]))]
    # columns = feature_names + vocabulary
    # columns.append("avg_value_length")

    if auto_clustering_enabled:
        clustering_results = KMeans(n_clusters=10, random_state=0).fit(col_features)
        feature_importance_result = feature_importance(10, feature_names, col_features)
        feature_importance_dict = pd.DataFrame(feature_importance_result)
        feature_importance_dict.to_csv('outputs/features.csv')
        col_labels_df = pd.DataFrame(col_features, columns=feature_names)
        col_labels_df['column_cluster_label'] = pd.DataFrame(clustering_results.labels_)
    else:
        col_labels_df = pd.DataFrame(col_features, columns=feature_names)
        ones_ = np.ones(len(col_features))
        col_labels_df['column_cluster_label'] = pd.DataFrame(ones_)

    number_of_clusters = len(col_labels_df['column_cluster_label'].unique())

    return col_labels_df, number_of_clusters

def feature_importance(n_clusters, ordered_feature_names, features):
    kms = KMeansInterp(
	n_clusters=n_clusters,
	ordered_feature_names= ordered_feature_names, 
	feature_importance_method='wcss_min', # or 'unsup2sup'
).fit(features)
    return kms.feature_importances_

def extract_labels(gt_path):
    filehandler = open(gt_path, "rb")
    labels_dict = pickle.load(filehandler)
    filehandler.close()
    return labels_dict

def get_number_of_clusters(col_groups_dir):
    number_of_col_clusters = 0
    for file in os.listdir(col_groups_dir):
        if ".pickle" in file:
            with open(os.path.join(col_groups_dir, file), 'rb') as filehandler:
                group_df = pickle.load(filehandler)
                number_of_clusters = len(group_df['column_cluster_label'].unique())
                number_of_col_clusters += number_of_clusters

    return number_of_col_clusters

def col_folding(context_df, sandbox_path, labels_path, col_groups_dir, auto_clustering_enabled, ner_model_name):
    clusters_dict = get_clusters_dict(context_df)
    if os.path.exists(col_groups_dir):
        shutil.rmtree(col_groups_dir)
    os.makedirs(col_groups_dir)
    logger.info("Col groups directory is created.")

    number_of_col_clusters = 0
    for cluster in clusters_dict:
        col_df = get_col_df(sandbox_path, clusters_dict[cluster], labels_path)
        col_features, feature_names = get_col_features(col_df, ner_model_name)
        col_labels_df, number_of_clusters = cluster_cols(col_features, auto_clustering_enabled, feature_names)
        number_of_col_clusters += number_of_clusters
        col_labels_df['col_value'] = col_df['col_value']
        col_labels_df['col_gt'] = col_df['col_gt']
        with open(os.path.join(col_groups_dir, "col_df_labels_cluster_{}.pickle".format(cluster)), "wb") \
                as filehandler:
            pickle.dump(col_labels_df, filehandler)
        col_labels_df[["column_cluster_label", "col_value"]].to_csv(
            os.path.join(col_groups_dir, "col_df_labels_cluster_{}.csv".format(cluster))
        )
    print("Finsihed")
    return number_of_col_clusters

