import logging
import math
import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

import app_logger
import generate_raha_features
from distributed import LocalCluster, Client
import xgboost as xgb
import dask.array as da
import sys


logger = logging.getLogger()


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def get_cells_features(sandbox_path, output_path):
    features_dict = dict()
    table_id = 0
    list_dirs_in_snd = os.listdir(sandbox_path)
    for parent in list_dirs_in_snd:
        table_dirs_path = os.path.join(sandbox_path, parent)
        table_dirs = os.listdir(table_dirs_path)
        for table in table_dirs:
            if not table.startswith("."):
                try:
                    path = os.path.join(table_dirs_path, table)
                    col_features, col_cell_values_list = generate_raha_features.generate_raha_features(table_dirs_path, table)
                    for col_idx in range(len(col_features)):
                        for row_idx in range(len(col_features[col_idx])):
                            features_dict[(table_id, col_idx, row_idx, 'og')] = np.append(col_features[col_idx][row_idx],
                                                                                        table_id)
                    dirty_df = pd.read_csv(path + "/dirty.csv", sep=",", header="infer", encoding="utf-8", dtype=str,
                                        keep_default_na=False, low_memory=False)
                    clean_df = pd.read_csv(path + "/" + table + ".csv", sep=",", header="infer", encoding="utf-8",
                                        dtype=str,
                                        keep_default_na=False, low_memory=False)
                    label_df = dirty_df.where(dirty_df.values == clean_df.values).notna() * 1
                    for col_idx, col_name in enumerate(label_df.columns):
                        for row_idx in range(len(label_df[col_name])):
                            features_dict[(table_id, col_idx, row_idx, 'gt')] = label_df[col_name][row_idx]
                    table_id += 1
                    logger.info("table_id: {}".format(table_id))
                except Exception as e:
                    logger.error(e)

    with open(os.path.join(output_path, "features.pkl"), "wb") as filehandler:
        pickle.dump(features_dict, filehandler)
    return features_dict


def sampling_labeling(x, y, n_cell_clusters_per_col_cluster, cells_clustering_alg):
    logger.info("sampling_labeling")
    clustering = None 

    if cells_clustering_alg == "km":
        clustering = KMeans(n_clusters=n_cell_clusters_per_col_cluster, random_state=0).fit(x)
    elif cells_clustering_alg == "hac":
        clustering = AgglomerativeClustering(n_clusters = n_cell_clusters_per_col_cluster).fit(x)

    cells_per_cluster = dict()
    labels_per_cluster = dict()

    for cell in enumerate(clustering.labels_):
        if cell[1] in cells_per_cluster.keys():
            cells_per_cluster[cell[1]].append(cell[0])
        else:
            cells_per_cluster[cell[1]] = [cell[0]]

    samples = []
    logger.info("labeling")

    shuffled_clusters = shuffle(list(cells_per_cluster.keys()))[:-1]
    for key in shuffled_clusters:
        sample = random.choice(cells_per_cluster[key])
        samples.append(sample)
        label = y[sample]
        labels_per_cluster[key] = label

    diff_n_clusters = n_cell_clusters_per_col_cluster - len(cells_per_cluster.keys())
    if diff_n_clusters != 0:
        logger.info("K-Means generated {} empty Clusters:))".format(diff_n_clusters))

    return cells_per_cluster, labels_per_cluster, samples


def get_train_test_sets(X_train, X_temp, y_train, y_temp, samples, cells_per_cluster, labels_per_cluster):
    logger.info("Label propagation")
    X_test, y_test = [], []
    for key in list(cells_per_cluster.keys()):
        for cell in cells_per_cluster[key]:
            if key in labels_per_cluster:
                X_train.append(X_temp[cell])
                y_train.append(labels_per_cluster[key])
            if cell not in samples:
                X_test.append(X_temp[cell])
                y_test.append(y_temp[cell])
    logger.info("Length of X_train: {}".format(len(X_train)))
    return X_train, y_train, X_test, y_test


def get_number_of_clusters(col_groups_dir):
    number_of_col_clusters = 0
    for file in os.listdir(col_groups_dir):
        if ".pickle" in file:
            with open(os.path.join(col_groups_dir, file), 'rb') as filehandler:
                group_df = pickle.load(filehandler)
                number_of_clusters = len(group_df['column_cluster_label'].unique())
                number_of_col_clusters += number_of_clusters

    return number_of_col_clusters


def error_detector(col_groups_dir, output_path, results_path, features_dict, n_labels, number_of_clusters, cell_clustering_alg):
    
    original_data_keys = []
    selected_samples = []
    predicted_all = []
    y_test_all = []
    X_labeled_by_user = []
    y_labeled_by_user = []

    n_cell_clusters_per_col_cluster = math.floor(n_labels / number_of_clusters) + 1
    n_cell_clusters_per_col_cluster_dict = {col_cluster: n_cell_clusters_per_col_cluster for col_cluster
                                            in range(number_of_clusters)}

    while sum(n_cell_clusters_per_col_cluster_dict.values()) < n_labels:
        rand = random.randint(0, number_of_clusters - 1)
        n_cell_clusters_per_col_cluster_dict[rand] += 1

    for file in os.listdir(col_groups_dir):
        if ".pickle" in file:
            file = open(os.path.join(col_groups_dir, file), 'rb')
            group_df = pickle.load(file)
            file.close()
            clusters = set(group_df['column_cluster_label'].sort_values())
            for c_idx, c in enumerate(clusters):
                X_train = []
                y_train = []
                X_temp = []
                y_temp = []
                original_data_keys_temp = []

                logger.info("Processing cluster {}, from {}".format(str(c), file))
                try:
                    c_df = group_df[group_df['column_cluster_label'] == c]
                    for index, row in c_df.iterrows():
                        for cell_idx in range(len(row['col_value'])):
                            original_data_keys_temp.append(
                                (row['table_id'], row['col_id'], cell_idx))
                            original_data_keys.append((row['table_id'], row['col_id'], cell_idx))
                            X_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'og')].tolist())
                            y_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'gt')].tolist())

                    cells_per_cluster, labels_per_cluster, samples = sampling_labeling(X_temp, y_temp,
                                                                        n_cell_clusters_per_col_cluster_dict[c_idx], cell_clustering_alg)
                    selected_samples += [original_data_keys_temp[sample] for sample in samples]
                    X_labeled_by_user.extend([X_temp[sample] for sample in samples])
                    y_labeled_by_user.extend([y_temp[sample] for sample in samples])

                    X_train, y_train, X_test, y_test = \
                            get_train_test_sets(X_train, X_temp, y_train, y_temp, 
                            samples, cells_per_cluster, labels_per_cluster)
                    predicted = classify(X_train, y_train, X_test)
                    predicted_all.extend(predicted)
                    y_test_all.extend(y_test) 
                    print("done")

                except Exception as e:
                    logger.error(e)
                    print("error", e)

    with open(os.path.join(output_path, "original_data_keys.pkl"), "wb") as filehandler:
        pickle.dump(original_data_keys, filehandler)

    with open(os.path.join(results_path, "sampled_tuples.pkl"), "wb") as filehandler:
        pickle.dump(selected_samples, filehandler)
        logger.info("Number of Labeled Cells: {}".format(len(selected_samples)))

    y_test_all.extend(y_labeled_by_user)
    predicted_all.extend(y_labeled_by_user)
    return y_test_all, predicted_all, original_data_keys, len(selected_samples)


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


def main(col_groups_files_path, output_path, results_path, features_dict, n_labels, number_of_clusters, cell_clustering_alg, classification_mode):
    y_test_all, predicted, original_data_keys, n_samples = error_detector(col_groups_files_path, output_path,
                                                                                 results_path,
                                                                                 features_dict, n_labels,
                                                                                 number_of_clusters, cell_clustering_alg)

    return y_test_all, predicted, original_data_keys, n_samples
