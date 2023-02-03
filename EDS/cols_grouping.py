import json
import logging
import math
import shutil
import os
import pickle
import csv
import numpy as np
import pandas as pd
import pip
from sklearn.cluster import MiniBatchKMeans

from messytables import CSVTableSet, type_guess
from kmeans_interp.kmeans_feature_imp import KMeansInterp
from statistics import median
from simhash import Simhash

logger = logging.getLogger()

type_dicts = {'Integer': 0, 'Decimal': 1, 'String': 2, 'Date': 3, 'Bool': 4, 'Time': 5, 'Currency': 6, 'Percentage': 7}


def specify_num_col_clusters(total_num_cells, total_labeling_budget, num_cols_tg, num_cells_tg):
    n_tg = math.floor(total_labeling_budget * num_cells_tg/total_num_cells)

    lambda_ = math.floor(n_tg/num_cols_tg)
    if lambda_>= 1:
        beta_tg = num_cols_tg
    else:
        # TODO: when we don't have enough budget to label all table groups
        beta_tg = math.ceil(num_cols_tg/n_tg)
    return beta_tg # num_col_clusters

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
    column_dict = {'table_path':[], 'table_id': [], 'col_id': [], 'col_value': [], 'col_gt': [], 'col_type': [], 'col_header': []}
    lake_labels_dict = extract_labels(labels_dict_path)
    total_num_cells_tg = 0

    for table in cluster:
        logging.info("Table: {}".format(table))
        parent_path = os.path.join(sandbox_path, table[1])
        table_path = os.path.join(parent_path, table[2] + "/dirty_clean.csv")
        df = pd.read_csv(table_path, quoting=csv.QUOTE_ALL)
        df = df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)

        table_file = open(table_path, 'rb')
        table_set = CSVTableSet(table_file)
        row_set = table_set.tables[0]
        types = type_guess(row_set.sample)

        for column_idx, column in enumerate(df.columns.tolist()):
            column_dict['table_path'].append(table_path)
            column_dict['table_id'].append(table[0])
            column_dict['col_id'].append(column_idx)
            column_dict['col_value'].append(df[column].tolist())
            column_dict['col_gt'].append(lake_labels_dict[(table[0], column_idx)].values)
            column_dict['col_type'].append(types[column_idx])
            column_dict['col_header'].append(column)
            total_num_cells_tg += len(df[column].tolist())

    col_df = pd.DataFrame.from_dict(column_dict)

    return col_df, total_num_cells_tg


def get_col_features(col_df):
    col_features = []
    characters_dictionary = {}
    tokens_dictionary = {}
    value_length_dictionary = {}
    count_char_tokens = {}

    feature_names = ['col_type']

    for i in range(col_df.shape[0]):
        features = []
        # profiles = dataset_profile(pd.DataFrame(col_df['col_value'][i]))
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

        count_char_tokens[i] = {'char':0, 'tokens': 0}
        value_length_sum = []
        for value in col_df['col_value'][i]:
            char_list = list(set(list(str(value))))
            if ' ' in char_list:
                char_list.remove(' ')
            count_char_tokens[i]['char'] += len(char_list)
            for character in char_list:
                if character not in characters_dictionary:
                    characters_dictionary[character] = {i: 0.0}
                if i not in characters_dictionary[character]:
                    characters_dictionary[character][i] =  0.0
                characters_dictionary[character][i] += 1.0
            value_length_sum.append(len(str(value)))
            tokens = str(value).split()
            if ' ' in tokens:
                tokens.remove(' ')
            count_char_tokens[i]['tokens'] += len(tokens)
            for token in tokens:
                if token not in tokens_dictionary:
                    tokens_dictionary[token] = {i: 0.0}
                if i not in tokens_dictionary[token]:
                    tokens_dictionary[token][i] =  0.0
                tokens_dictionary[token][i] += 1.0

        value_length_dictionary[i] = value_length_sum

    for key in value_length_dictionary.keys():
        # sum_value_length_dictionary[key] = sum_value_length_dictionary[key]/len(col_df['col_value'][key])
        value_length_dictionary[key] = median(value_length_dictionary[key])

    for token in list(tokens_dictionary.keys()):
        if token in characters_dictionary.keys():
            del tokens_dictionary[token]


    for i in range(col_df.shape[0]):
        
        column_profile = {"characters": dict(), "tokens": dict(), "median_value_length": 0}

        for ch in list(characters_dictionary.keys()):
            if i in characters_dictionary[ch]:
                column_profile["characters"][ch] = characters_dictionary[ch][i] / len(col_df['col_value'][i])
            else:
                column_profile["characters"][ch] = 0

        for t in list(tokens_dictionary.keys()):
            if i in tokens_dictionary[t]:
                column_profile["tokens"][t] = tokens_dictionary[t][i] / len(col_df['col_value'][i])
            else:
                column_profile["tokens"][t] = 0

        column_profile["median_value_length"] = value_length_dictionary[i]
        
        char_list = list(column_profile["characters"].values())
        tokens_list = list(column_profile["tokens"].values())

        for char in char_list:
            col_features[i].append(char)
            
        for token in tokens_list:
            col_features[i].append(token)

        col_features[i].append(column_profile['median_value_length'])
        col_features[i].append(Simhash(col_df['col_header'][i]).value)

    feature_names.extend([c for c in characters_dictionary.keys()])
    feature_names.extend([str(v) for v in tokens_dictionary.keys()])
    feature_names.append("median_value_length")
    feature_names.append("header")

    return col_features, feature_names

def cluster_cols(col_features, clustering_enabled, feature_names, beta_tg):

    if clustering_enabled:
        logging.info("Clustering columns")
        # TODO memory profiler 
        clustering_results = MiniBatchKMeans(n_clusters=12, random_state=0, reassignment_ratio=0, batch_size = 256*64).fit(col_features)        
        # TODO: evaluation 
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

def get_cluster_size(col_labels_df):
    col_cluster_sizes = dict()
    for col_cluster in col_labels_df["column_cluster_label"].unique():
        col_cluster_sizes[str(col_cluster)] = 0
        clusters_df = col_labels_df.loc[col_labels_df["column_cluster_label"]==col_cluster]
        for idx, row in clusters_df.iterrows():
            col_cluster_sizes[str(col_cluster)] += len(row["col_value"])     
    return col_cluster_sizes

def col_folding(total_num_cells, total_labeling_budget, context_df, sandbox_path, labels_path, col_groups_dir, clustering_enabled):
    clusters_dict = get_clusters_dict(context_df)
    if os.path.exists(col_groups_dir):
        shutil.rmtree(col_groups_dir)
    os.makedirs(col_groups_dir)
    logger.info("Col groups directory is created.")
    number_of_col_clusters = dict()
    cluster_sizes = dict()
    for cluster in clusters_dict:
        col_df, total_num_cells_tg = get_col_df(sandbox_path, clusters_dict[cluster], labels_path)
        beta_tg = specify_num_col_clusters(total_num_cells, total_labeling_budget, col_df.shape[0], total_num_cells_tg)
        logging.info("beta_tg: {}".format(beta_tg))
        logging.info("cluster: {}".format(cluster))
        col_features, feature_names = get_col_features(col_df)
        col_labels_df, number_of_clusters = cluster_cols(col_features, clustering_enabled, feature_names, beta_tg)
        number_of_col_clusters[str(cluster)] = number_of_clusters
        col_labels_df['col_value'] = col_df['col_value']
        col_labels_df['col_chars'] = col_df['col_value'].apply(lambda x: set([ch for val in x for ch in str(val)]))
        col_labels_df['col_gt'] = col_df['col_gt']
        col_labels_df['table_id'] = col_df['table_id']
        col_labels_df['col_id'] = col_df['col_id']
        col_labels_df['table_path'] = col_df['table_path']
        col_labels_df['table_cluster'] = str(cluster)

        cluster_sizes[str(cluster)] = get_cluster_size(col_labels_df)
        with open(os.path.join(col_groups_dir, "col_df_labels_cluster_{}.pickle".format(cluster)), "wb") \
                as filehandler:
            pickle.dump(col_labels_df, filehandler)
        col_labels_df[["column_cluster_label", "col_value", "table_id", "table_path", 'table_cluster', "col_id"]].to_csv(
            os.path.join(col_groups_dir, "col_df_labels_cluster_{}.csv".format(cluster))
        )
    with open(os.path.join(col_groups_dir, "cluster_sizes_all.json"), "w", encoding="utf8") \
                as filehandler:
                filehandler.write(json.dumps(cluster_sizes))
    with open(os.path.join(col_groups_dir, "number_of_col_clusters.json"), "w", encoding="utf8") \
                as filehandler:
                filehandler.write(json.dumps(number_of_col_clusters))
    logger.info("Column Grouping Finished")
    return number_of_col_clusters, cluster_sizes

