import collections
import hashlib
import html
import json
import logging
import math
import os
import pickle
import random
import re
import numpy as np
import pandas as pd
import scipy
from sklearn import manifold
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle

import app_logger
import generate_raha_features
from distributed import LocalCluster, Client
import xgboost as xgb
import dask.array as da
import sys
from statistics import mode
import ed_twolevel_rahas_features
import fastcluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold

from read_data import read_csv
from scipy.spatial.distance import pdist, squareform
import numpy as np


logger = logging.getLogger()


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")



def get_cells_features(sandbox_path, output_path, table_char_set_dict, tables_dict):

    features_dict = dict()
    list_dirs_in_snd = os.listdir(sandbox_path)
    list_dirs_in_snd.sort()
    for parent in list_dirs_in_snd:
        table_dirs_path = os.path.join(sandbox_path, parent)
        table_dirs = os.listdir(table_dirs_path)
        table_dirs.sort()
        for table in table_dirs:
            if not table.startswith("."):
                print("************************table: ", table)
                try:
                    path = os.path.join(table_dirs_path, table)
                    table_file_name_santos = tables_dict[table]

                    dirty_df = read_csv(os.path.join(path, "dirty_clean.csv"), low_memory=False)
                    clean_df = read_csv(os.path.join(path + "/clean.csv"), low_memory=False)

                    # TODO
                    
                    logging.info("Generating features for table: " + table)
                    charsets = dict()
                    for idx, col in enumerate(dirty_df.columns):
                        # charsets[idx] = table_char_set_dict[(str(table_id), str(idx))]
                        charsets[idx] = table_char_set_dict[(str(hashlib.md5(table_file_name_santos.encode()).hexdigest()), str(idx))]
                    print("generate features ---- table: ", table)
                    col_features = generate_raha_features.generate_raha_features(table_dirs_path, table, charsets)
                    print("generate features done ---- table: ", table)
                    for col_idx in range(len(col_features)):
                        for row_idx in range(len(col_features[col_idx])):
                            # table_id_added = np.append(col_features[col_idx][row_idx], table_id)
                            #col_idx_added = np.append(table_id_added, col_idx)
                            features_dict[(hashlib.md5(table_file_name_santos.encode()).hexdigest(), col_idx, row_idx, 'og')] = col_features[col_idx][row_idx]
                            
                    
                    label_df = dirty_df.where(dirty_df.values != clean_df.values).notna() * 1
                    for col_idx, col_name in enumerate(label_df.columns):
                        for row_idx in range(len(label_df[col_name])):
                            features_dict[(hashlib.md5(table_file_name_santos.encode()).hexdigest(), col_idx, row_idx, 'gt')] = label_df[col_name][row_idx]
                    logger.info("table: {}".format(table))
                except Exception as e:
                    logger.error(e)


    with open(os.path.join(output_path, "features.pickle"), "wb") as filehandler:
        pickle.dump(features_dict, filehandler)
    return features_dict


def sampling_labeling_iso(table_cluster, col_cluster, x, y, n_cell_clusters_per_col_cluster, cells_clustering_alg, value_temp):
    
    logger.info("Sampling and labeling")

    ""
    model = IsolationForest()
    model.fit(x)
    iso_for_labels = model.predict(x)
    outliers = []
    outliers_orig_x_idx = dict()
    non_outliers = []
    for i, value in enumerate(iso_for_labels):
        if value == -1:
            outliers.append(x[i])
            outliers_orig_x_idx[len(outliers) - 1] = i
        else:
            non_outliers.append(i)
    non_outliers_features = []
    for i, value in enumerate(non_outliers):
        non_outliers_features.append(x[value])
    non_outliers_center = np.mean(non_outliers_features, axis=0)

    
    iso_for_scores = model.decision_function(x)

    # Find the indices of the negative values in iso_for_scores
    neg_indices = np.where(iso_for_scores < 0)[0]

    # Sort the negative values in ascending order
    neg_sorted_indices = neg_indices[np.argsort(iso_for_scores[neg_indices])]

    # Find the index corresponding to the 0.01 percentile of the negative values
    index_0_01_percentile = neg_sorted_indices[int(0.05 * len(neg_sorted_indices))]

    data_points_weights = []
    for i, value in enumerate(iso_for_scores):
        if i in neg_sorted_indices[:index_0_01_percentile]:
            data_points_weights.append(1)
        else:
            data_points_weights.append(0.000001)
    
    clustering = None 

    if cells_clustering_alg == "km":
        n_cell_clusters_per_col_cluster = min(len(outliers),n_cell_clusters_per_col_cluster)
        logging.info("KMeans - n_cell_clusters_per_col_cluster: {}".format(n_cell_clusters_per_col_cluster))
        if len(x) < n_cell_clusters_per_col_cluster + 1:
            n_cell_clusters_per_col_cluster = len(x) - 1 
        clustering = MiniBatchKMeans(n_clusters= n_cell_clusters_per_col_cluster, random_state=0, reassignment_ratio=0, batch_size = 256 * 64).fit(outliers)
        logging.info("KMeans - n_cell_clusters_generated: {}".format(len(set(clustering.labels_))))
        clustering_labels = clustering.labels_
        
        
    logging.info("cells per cluster")
    cells_per_cluster = dict()
    # TODO remove this
    errors_per_cluster = dict()
    labels_per_cluster_all = dict()
    labels_per_cluster = dict()

    for cell in enumerate(clustering_labels):
        orig_x_idx = outliers_orig_x_idx[cell[0]]
        if cell[1] in cells_per_cluster.keys():
            cells_per_cluster[cell[1]].append(orig_x_idx)
            if y[orig_x_idx] == 1:
                errors_per_cluster[cell[1]] += 1
        else:
            cells_per_cluster[cell[1]] = [orig_x_idx]
            labels_per_cluster_all[cell[1]] = []
            errors_per_cluster[cell[1]] = y[orig_x_idx]
        # logging.info("**************** error ratio per cluster: {}".format(errors_per_cluster[cell[1]]/ len(cells_per_cluster[cell[1]])))
    cells_per_cluster[len(cells_per_cluster.keys())] = non_outliers
    samples = []
    sample_values = []
    samples_orig_values = []

    if cells_clustering_alg == "km":
        for cluster in list(cells_per_cluster.keys())[:-1]:
            # weights_cluster_points = [data_points_weights[i] for i in cells_per_cluster[cluster]]
            # total_weight = sum(weights_cluster_points)
            # normalized_weights = [w / total_weight for w in weights_cluster_points]
            # samples_tmp = np.random.choice(cells_per_cluster[cluster], size=3)
            # compute anomaly scores (distance based)
            outliers_dist = []
            for i in cells_per_cluster[cluster]:
                value = x[i]
                outliers_dist.append(np.linalg.norm(value - non_outliers_center))
            samples_tmp_out_idx = outliers_dist.index(max(outliers_dist))
            samples_tmp = [cells_per_cluster[cluster][samples_tmp_out_idx]]
            samples.extend(samples_tmp) # sample k vectors
            sample_values.extend([x[i] for i in samples_tmp]) # get the selected vectors
            samples_orig_values.extend([value_temp[i] for i in samples_tmp]) # get the selected vectors values

    logger.info("labeling")
    for cell in enumerate(clustering_labels):
        orig_x_idx = outliers_orig_x_idx[cell[0]]
        if orig_x_idx in samples:
            labels_per_cluster_all[cell[1]].append(y[orig_x_idx])
    
    for c in labels_per_cluster_all.keys():
        if len(labels_per_cluster_all[c]) > 0:
            labels_per_cluster[c] = mode(labels_per_cluster_all[c])  

    universal_samples = dict()
    for s in samples:
        universal_samples[(table_cluster, col_cluster, s)] = y[s]

    # logger.info("labels_per_cluster: {}".format(labels_per_cluster))
    # logger.info("samples: {}".format(samples_orig_values))
    return cells_per_cluster, labels_per_cluster, universal_samples, samples

def get_train_test_sets(X_temp, y_temp, samples, cells_per_cluster, labels_per_cluster):
    logger.info("Train-Test set preparation")
    X_train, y_train, X_test, y_test, y_cell_ids = [], [], [], [], []
    clusters = list(cells_per_cluster.keys())
    clusters.sort()
    for key in clusters:
        for cell in cells_per_cluster[key]:
            if key in labels_per_cluster:
                X_train.append(X_temp[cell])
                y_train.append(labels_per_cluster[key])
            if cell not in samples:
                X_test.append(X_temp[cell])
                y_test.append(y_temp[cell])
                y_cell_ids.append(cell)
    logger.info("Length of X_train: {}".format(len(X_train)))
    return X_train, y_train, X_test, y_test, y_cell_ids


def sampling_labeling(table_cluster, col_cluster, x, y, keys, n_cell_clusters_per_col_cluster, cells_clustering_alg, value_temp, noise_status):
    
    logger.info("Sampling and labeling")

    ""
    
    outliers = []
    outliers_orig_x_idx = dict()
    non_outliers = []
    non_outliers_features = []

    for i, key in enumerate(keys):
        if noise_status[key] == True:
            outliers.append(x[i])
            outliers_orig_x_idx[len(outliers) - 1] = i
        else:
            non_outliers.append(i)
            non_outliers_features.append(x[i])
        
    non_outliers_center = np.mean(non_outliers_features, axis=0)

    if len(outliers) == 0 or len(non_outliers) / len(non_outliers) < 0.1:
        return None, None, None, None
        
    clustering = None 

    if cells_clustering_alg == "km":
        n_cell_clusters_per_col_cluster = min(len(outliers),n_cell_clusters_per_col_cluster)
        logging.info("KMeans - n_cell_clusters_per_col_cluster: {}".format(n_cell_clusters_per_col_cluster))
        if len(x) < n_cell_clusters_per_col_cluster + 1:
            n_cell_clusters_per_col_cluster = len(x) - 1 
        clustering = MiniBatchKMeans(n_clusters= n_cell_clusters_per_col_cluster, random_state=0, reassignment_ratio=0, batch_size = 256 * 64).fit(outliers)
        logging.info("KMeans - n_cell_clusters_generated: {}".format(len(set(clustering.labels_))))
        clustering_labels = clustering.labels_
        
        
    logging.info("cells per cluster")
    cells_per_cluster = dict()
    # TODO remove this
    errors_per_cluster = dict()
    labels_per_cluster_all = dict()
    labels_per_cluster = dict()

    for cell in enumerate(clustering_labels):
        orig_x_idx = outliers_orig_x_idx[cell[0]]
        if cell[1] in cells_per_cluster.keys():
            cells_per_cluster[cell[1]].append(orig_x_idx)
            if y[orig_x_idx] == 1:
                errors_per_cluster[cell[1]] += 1
        else:
            cells_per_cluster[cell[1]] = [orig_x_idx]
            labels_per_cluster_all[cell[1]] = []
            errors_per_cluster[cell[1]] = y[orig_x_idx]
        # logging.info("**************** error ratio per cluster: {}".format(errors_per_cluster[cell[1]]/ len(cells_per_cluster[cell[1]])))
    cells_per_cluster[len(cells_per_cluster.keys())] = non_outliers
    samples = []
    sample_values = []
    samples_orig_values = []

    if cells_clustering_alg == "km":
        for cluster in list(cells_per_cluster.keys())[:-1]:
            # weights_cluster_points = [data_points_weights[i] for i in cells_per_cluster[cluster]]
            # total_weight = sum(weights_cluster_points)
            # normalized_weights = [w / total_weight for w in weights_cluster_points]
            # samples_tmp = np.random.choice(cells_per_cluster[cluster], size=3)
            # compute anomaly scores (distance based)
            outliers_dist = []
            for i in cells_per_cluster[cluster]:
                value = x[i]
                outliers_dist.append(np.linalg.norm(value - non_outliers_center))
            samples_tmp_out_idx = outliers_dist.index(max(outliers_dist))
            samples_tmp = [cells_per_cluster[cluster][samples_tmp_out_idx]]
            samples.extend(samples_tmp) # sample k vectors
            sample_values.extend([x[i] for i in samples_tmp]) # get the selected vectors
            samples_orig_values.extend([value_temp[i] for i in samples_tmp]) # get the selected vectors values

    logger.info("labeling")
    for cell in enumerate(clustering_labels):
        orig_x_idx = outliers_orig_x_idx[cell[0]]
        if orig_x_idx in samples:
            labels_per_cluster_all[cell[1]].append(y[orig_x_idx])
    
    for c in labels_per_cluster_all.keys():
        if len(labels_per_cluster_all[c]) > 0:
            labels_per_cluster[c] = mode(labels_per_cluster_all[c])  

    universal_samples = dict()
    for s in samples:
        universal_samples[(table_cluster, col_cluster, s)] = y[s]

    # logger.info("labels_per_cluster: {}".format(labels_per_cluster))
    # logger.info("samples: {}".format(samples_orig_values))
    return cells_per_cluster, labels_per_cluster, universal_samples, samples



def get_n_cell_clusters_per_col_cluster_dict(n_labels, cluster_sizes, number_of_col_clusters):
    number_of_all_col_clusters = sum(number_of_col_clusters.values())
    assigned_labels = 0
    init_n_labels = n_labels
    n_cells = 0
    for table_cluster in cluster_sizes.keys():
        for col_cluster in cluster_sizes[table_cluster].keys():
            n_cells += cluster_sizes[table_cluster][col_cluster]

    n_cell_clusters_per_col_cluster_dict = dict()

    for table_cluster in cluster_sizes.keys():
        for col_cluster in cluster_sizes[table_cluster].keys():
             n_cell_clusters_per_col_cluster_dict[(table_cluster, col_cluster)] = \
                max(2, math.floor(init_n_labels * cluster_sizes[table_cluster][col_cluster] / n_cells))

    assigned_labels = sum(n_cell_clusters_per_col_cluster_dict.values())
    while assigned_labels > init_n_labels:
        rand = random.choice(list(n_cell_clusters_per_col_cluster_dict.keys()))
        if n_cell_clusters_per_col_cluster_dict[rand] > 2:
            n_cell_clusters_per_col_cluster_dict[rand] -= 1
            assigned_labels -= 1

    while assigned_labels < init_n_labels:
        remaining_labels = init_n_labels - assigned_labels
        n_cell_clusters_per_col_cluster = math.ceil(remaining_labels / number_of_all_col_clusters)
        for table_cluster in cluster_sizes.keys():
            for col_cluster in cluster_sizes[table_cluster].keys():
                current_labels = n_cell_clusters_per_col_cluster_dict[(table_cluster, col_cluster)]
                max_labels = cluster_sizes[table_cluster][col_cluster]
                if current_labels + n_cell_clusters_per_col_cluster <= max_labels:
                    n_assigned_labels = current_labels + n_cell_clusters_per_col_cluster
                else:
                    n_assigned_labels = max_labels
                n_cell_clusters_per_col_cluster_dict[(table_cluster, col_cluster)] = n_assigned_labels
                assigned_labels += (n_assigned_labels - current_labels)
                if assigned_labels >= init_n_labels:
                    break
            if assigned_labels >= init_n_labels:
                break
    return n_cell_clusters_per_col_cluster_dict

def get_col_cluster_noises(table_cluster, group_df, features_dict, noises_dict):
    logger.info("get_col_cluster_noises")
    n_col_clusters = len(group_df['column_cluster_label'].unique())
    for cluster in range(n_col_clusters):
        X_temp = []
        y_temp = []
        X_temp_dict = dict()
        noise_status = dict()
        c_df = group_df[group_df['column_cluster_label'] == cluster]
        for index, row in c_df.iterrows():
            for cell_idx in range(len(row['col_value'])):
                X_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'og')].tolist())
                y_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'gt')])
                X_temp_dict[len(X_temp) - 1] = (row['table_id'], row['col_id'], cell_idx)
            
        clustering = DBSCAN(min_samples = 2).fit(X_temp)
        labels = clustering.labels_
        cells_per_db_clusters = dict()
        for i in range(len(labels)):
            if labels[i] not in cells_per_db_clusters:
                cells_per_db_clusters[labels[i]] = []
            cells_per_db_clusters[labels[i]].append(X_temp_dict[i])
        error_ratio = dict()
        for key in cells_per_db_clusters.keys():
            if key == -1:
                continue
            error_ratio[key] = 0
            for cell in cells_per_db_clusters[key]:
                if features_dict[(cell[0], cell[1], cell[2], 'gt')] == 1:
                    error_ratio[key] += 1
            if y_temp.count(1) > 0:
                error_ratio[key] /= y_temp.count(1)
            else:
                error_ratio[key] = None
        noise_count = 0
        for l in labels:
            if l == -1:
                noise_count += 1
        for i in range(len(labels)):
            if labels[i] == -1:
                noise_status[X_temp_dict[i]] = True
            else:
                noise_status[X_temp_dict[i]] = False
        noises_dict["table_cluster"].append(table_cluster)
        noises_dict["col_cluster"].append(cluster)
        noises_dict["n_noises"].append(noise_count)
        noises_dict["noise_status"].append(noise_status)
        noises_dict["error_ratio"].append(error_ratio)
        logging.info("DBSCAN table cluster {} col cluster {} done - n_noises: {}".format(table_cluster, cluster, noise_count))
            
    return noises_dict

def process_col_cluster(n_cell_clusters_per_col_cluster, table_cluster, cluster,\
                         group_df, features_dict, cell_clustering_alg, noise_status, df_n_labels):
    X_train = []
    y_train = []
    X_temp = []
    y_temp = []
    key_temp = []
    noise_temp = []
    value_temp = []
    original_data_keys_temp = []
    X_labeled_by_user = []
    y_labeled_by_user = []

    current_local_cell_uid = 0
    datacells_uids = dict()

    logger.info("Processing cluster {}".format(str(cluster)))

    try:
        c_df = group_df[group_df['column_cluster_label'] == cluster]
        for index, row in c_df.iterrows():
            for cell_idx in range(len(row['col_value'])):
                original_data_keys_temp.append(
                    (row['table_id'], row['col_id'], cell_idx, row['col_value'][cell_idx]))

                value_temp.append(row['col_value'][cell_idx])
                X_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'og')].tolist())
                y_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'gt')].tolist())
                key_temp.append((row['table_id'], row['col_id'], cell_idx))
                noise_temp.append(noise_status[(row['table_id'], row['col_id'], cell_idx)])
                datacells_uids[(row['table_id'], row['col_id'], cell_idx, row['col_value'][cell_idx])] = current_local_cell_uid
                current_local_cell_uid += 1
        
        tp = []
        fp = []

        for i, v in enumerate(y_temp):
            if v == 1 and noise_temp[i] == True:
                tp.append(i)
            elif v == 0 and noise_temp[i] == True:
                fp.append(i)

        n_errs = y_temp.count(1)
        logging.info("--------------------Noises - DBSCAN INFO--------------------")
        if n_errs == 0:
            logging.info("Table_cluster: {}, cluster: {}, detcetd errors".format(table_cluster, cluster, len(tp)))
        else:
            logging.info("Table_cluster: {}, cluster: {}, truely detcetd errors/all real errors: {}, truely detected errors / all detected errors: {}, wrongly detected errors / all detected errors:{}".format(table_cluster, cluster, len(tp)/ y_temp.count(1), len(tp)/ noise_temp.count(True) if noise_temp.count(True) > 0 else 0, len(fp)/ noise_temp.count(True) if noise_temp.count(True) > 0 else 0))
        logging.info("-----------------------------------------------------------")
        model = IsolationForest()
        model.fit(X_temp)
        y_pred = model.predict(X_temp).tolist()
        tp = []
        fp = []
        for i, v in enumerate(y_pred):
            if v == -1 and y_temp[i] == 1:
                tp.append(i)
            elif v == -1 and y_temp[i] == 0:
                fp.append(i)
        logging.info("***************************Outliers - ISO INFO***************************")
        if n_errs == 0:
            logging.info("No Errors")
            logging.info("Table_cluster: {}, cluster: {}, detcetd errors:{}".format(table_cluster, cluster, len(tp)))
        else:
            logging.info("Table_cluster: {}, cluster: {}, truely detcetd errors/all real errors: {}, truely detected errors / all detected errors: {}, wrongly detected errors / all detected errors:{}".format(table_cluster, cluster, len(tp)/ y_temp.count(1), len(tp)/ y_pred.count(-1) if y_pred.count(-1) > 0 else 0, len(fp)/ y_pred.count(-1) if y_pred.count(-1) > 0 else 0))
        
        cells_per_cluster, labels_per_cluster, universal_samples, samples = sampling_labeling_iso(table_cluster, cluster, X_temp, y_temp,
                                                                                                  n_cell_clusters_per_col_cluster, cell_clustering_alg, value_temp)
        
        # cells_per_cluster, labels_per_cluster, universal_samples, samples = sampling_labeling_iso(table_cluster, cluster, X_temp, y_temp, key_temp,
                                                            # n_cell_clusters_per_col_cluster, cell_clustering_alg, value_temp, noise_status)
        
        if samples is None:
            return None, None, None, None, None, None, None, None 
        X_labeled_by_user.extend([X_temp[sample] for sample in samples])
        y_labeled_by_user.extend([y_temp[sample] for sample in samples])

        X_train, y_train, X_test, y_test, y_cell_ids = \
                get_train_test_sets(X_temp, y_temp, samples, cells_per_cluster, labels_per_cluster)
        predicted = classify(X_train, y_train, X_test)

    except Exception as e:
        logger.error(e)
        logging.error("error", e)
    return y_test, y_cell_ids, predicted, original_data_keys_temp, universal_samples, X_labeled_by_user, y_labeled_by_user, datacells_uids

def get_n_cell_groups_noise_based(noises_dict, cluster_sizes, labeling_budget):
    #TODO: datatype noises_dict
    df = pd.DataFrame.from_dict(noises_dict).sort_values(by=['n_noises'], ascending=False)
    cluster_sizes_df = pd.DataFrame.from_dict(cluster_sizes)
    size_dict = {"table_cluster": [], "col_cluster": [], "n_cells": []}
    for tab_cl in cluster_sizes:
        for col_cl in cluster_sizes[tab_cl]:
            size_dict["table_cluster"].append(tab_cl)
            size_dict["col_cluster"].append(int(col_cl))
            size_dict["n_cells"].append(cluster_sizes_df[tab_cl][col_cl])

    df = pd.merge(df, pd.DataFrame.from_dict(size_dict), on = ['table_cluster', 'col_cluster'])

    df["noise_ratio"] = df["n_noises"] / df["n_cells"]
    df = df.sort_values(by=['noise_ratio'], ascending=False)
    df["n_cell_groups"] = df.apply(lambda x: min(x["n_cells"], 2), axis=1)
    used_labels = 2 * df.shape[0]
    if labeling_budget > used_labels:
        sum_ratio = df["noise_ratio"].sum()
        df["n_cell_groups"] = \
            df.apply(lambda x: x["n_cell_groups"] + math.floor(min(x["n_cells"] - x["n_cell_groups"], x["noise_ratio"]/sum_ratio * (labeling_budget - used_labels))), axis=1)

    return df

def error_detector(cell_feature_generator_enabled, noise_extraction_enabled, sandbox_path, col_groups_dir, 
                   output_path, results_path, n_labels, number_of_col_clusters, 
                   cluster_sizes, cell_clustering_alg, tables_dict):

    logging.info("Starting error detection")
    original_data_keys = []
    unique_cells_local_index_collection = dict()
    predicted_all = dict()
    y_test_all = dict()
    y_local_cell_ids = dict()
    X_labeled_by_user_all = dict()
    y_labeled_by_user_all = dict()
    char_set_dict = dict()
    table_charset_dict = dict()
    selected_samples = dict()

    for file_name in os.listdir(col_groups_dir):
        if ".pickle" in file_name:
            file = open(os.path.join(col_groups_dir, file_name), 'rb')
            group_df = pickle.load(file)
            if not isinstance(group_df, pd.DataFrame):
                group_df = pd.DataFrame.from_dict(group_df, orient='index').T
            table_cluster = group_df['table_cluster'].values[0]
            file.close()
            clusters = set(group_df['column_cluster_label'].sort_values())
            for c_idx, cluster in enumerate(clusters):
                charset_val = []
                for idx, row in group_df[group_df['column_cluster_label'] == cluster].iterrows():
                    row = [str(val) for val in row['col_value']]
                    charset_val.extend(''.join(row))
                charset_val = set(charset_val)
                char_set_dict[(str(table_cluster), str(cluster))] = charset_val

            table_ids = set(group_df['table_id'].values)
            for table_id in table_ids:
                col_ids = set(group_df[group_df['table_id'] == table_id]['col_id'].values)
                for col_id in col_ids:
                    table_charset_dict[(str(table_id), str(col_id))] = char_set_dict[str(table_cluster), str(group_df[(group_df['table_id'] == table_id) & (group_df['col_id'] == col_id)]['column_cluster_label'].values[0])]
            logger.info("Charset dictionary generated.")

    if cell_feature_generator_enabled:
        features_dict = ed_twolevel_rahas_features.get_cells_features(sandbox_path, output_path, table_charset_dict, tables_dict)
        logger.info("Generating cell features started.")
    else:
        with open(os.path.join(output_path, "features.pickle"), 'rb') as pickle_file:
            features_dict = pickle.load(pickle_file)

    # Extracting Noises 
    if noise_extraction_enabled:
        noises_dict = {"table_cluster": [], "col_cluster": [], "n_noises": [], "noise_status":[], "error_ratio":[]}
        for file_name in os.listdir(col_groups_dir):
            if ".pickle" in file_name:
                file = open(os.path.join(col_groups_dir, file_name), 'rb')
                group_df = pickle.load(file)
                if not isinstance(group_df, pd.DataFrame):
                    group_df = pd.DataFrame.from_dict(group_df, orient='index').T
                table_cluster = file_name.removeprefix("col_df_labels_cluster_").removesuffix(".pickle")
                file.close()
                noises_dict = get_col_cluster_noises(table_cluster, group_df, features_dict, noises_dict)
        with open(os.path.join(output_path, "noises_dict.pickle"), "wb") as filehandler:
            pickle.dump(noises_dict, filehandler)
        print("Noise extraction finished.")
        
    else:
         with open(os.path.join(output_path, "noises_dict.pickle"), 'rb') as pickle_file:
            noises_dict = pickle.load(pickle_file)
    df_n_labels = get_n_cell_groups_noise_based(noises_dict, cluster_sizes, labeling_budget=n_labels)


    for file_name in os.listdir(col_groups_dir):
        if ".pickle" in file_name:
            file = open(os.path.join(col_groups_dir, file_name), 'rb')
            group_df = pickle.load(file)
            if not isinstance(group_df, pd.DataFrame):
                group_df = pd.DataFrame.from_dict(group_df, orient='index').T
            table_cluster = file_name.removeprefix("col_df_labels_cluster_").removesuffix(".pickle")
            file.close()
            clusters = set(group_df['column_cluster_label'].sort_values())
            for c_idx, cluster in enumerate(clusters):
                n_cell_groups = df_n_labels[(df_n_labels['table_cluster'] == table_cluster) & (df_n_labels['col_cluster'] == cluster)]['n_cell_groups'].values[0]
                noise_status = df_n_labels[(df_n_labels['table_cluster'] == table_cluster) & (df_n_labels['col_cluster'] == cluster)]['noise_status'].values[0]
                y_test, y_cell_ids, predicted, original_data_keys_temp, universal_samples, \
                X_labeled_by_user, y_labeled_by_user, datacells_local_ids = \
                    process_col_cluster(n_cell_groups, table_cluster, cluster, group_df, features_dict, cell_clustering_alg, noise_status, df_n_labels)
                if X_labeled_by_user is not None:
                    selected_samples.update(universal_samples)
                    original_data_keys.extend(original_data_keys_temp)

                    X_labeled_by_user_all[(str(table_cluster), str(cluster))] = X_labeled_by_user
                    y_labeled_by_user_all[(str(table_cluster), str(cluster))] = y_labeled_by_user
                    
                    predicted_all[(str(table_cluster), str(cluster))] = predicted
                    y_test_all[(str(table_cluster), str(cluster))] = y_test
                    y_local_cell_ids[(str(table_cluster), str(cluster))] = y_cell_ids
                    unique_cells_local_index_collection[(str(table_cluster), str(cluster))] = datacells_local_ids

                logging.info("done - Processing col cluster {} table cluster {}".format(str(cluster), str(table_cluster)))

    with open(os.path.join(output_path, "original_data_keys.pkl"), "wb") as filehandler:
        pickle.dump(original_data_keys, filehandler)

    with open(os.path.join(results_path, "sampled_tuples.pkl"), "wb") as filehandler:
        pickle.dump(selected_samples, filehandler)
        logger.info("Number of Labeled Cells: {}".format(len(selected_samples)))

    return y_test_all, y_local_cell_ids, predicted_all, y_labeled_by_user_all,\
                unique_cells_local_index_collection, selected_samples
    


def classify(X_train, y_train, X_test):
    logger.info("Classification")
    if sum(y_train) == 0:
        predicted = [0] * len(X_test)
    elif sum(y_train) == len(y_train):
        predicted = [1] * len(X_test)
    else:
        imp = SimpleImputer(strategy="most_frequent")
        gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        clf = make_pipeline(imp, gbc)
        clf.fit(np.asarray(X_train), np.asarray(y_train))
        predicted = clf.predict(X_test)
    return predicted

