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

    for key in cells_per_cluster.keys():
        sample = random.choice(cells_per_cluster[key])
        samples.append(sample)
        label = y[sample]
        labels_per_cluster[key] = label

    diff_n_clusters = n_cell_clusters_per_col_cluster - len(cells_per_cluster.keys())
    if diff_n_clusters != 0:
        logger.info("K-Means generated {} empty Clusters:))".format(diff_n_clusters))

    return cells_per_cluster, labels_per_cluster, samples


def label_propagation(x_train, x_tmp, y_train, cells_per_cluster, labels_per_cluster):
    logger.info("Label propagation")
    for key in list(cells_per_cluster.keys()):
        for cell in cells_per_cluster[key]:
            x_train.append(x_tmp[cell])
            y_train.append(labels_per_cluster[key])
    logger.info("Length of X_train: {}".format(len(x_train)))
    return x_train, y_train


def get_number_of_clusters(col_groups_dir):
    number_of_col_clusters = 0
    for file in os.listdir(col_groups_dir):
        if ".pickle" in file:
            with open(os.path.join(col_groups_dir, file), 'rb') as filehandler:
                group_df = pickle.load(filehandler)
                number_of_clusters = len(group_df['column_cluster_label'].unique())
                number_of_col_clusters += number_of_clusters

    return number_of_col_clusters


def get_train_test_sets(col_groups_dir, output_path, results_path, features_dict, n_labels, number_of_clusters, cell_clustering_alg):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    original_data_values = []
    labels = []

    n_cell_clusters_per_col_cluster = math.floor(n_labels / number_of_clusters)
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
                logger.info("Processing cluster {}, from {}".format(str(c), file))
                try:
                    X_tmp = []
                    y_tmp = []
                    original_data_values_tmp = []
                    c_df = group_df[group_df['column_cluster_label'] == c]
                    for index, row in c_df.iterrows():
                        for cell_idx in range(len(row['col_value'])):
                            X_test.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'og')].tolist())
                            y_test.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'gt')].tolist())
                            original_data_values.append(
                                (row['table_id'], row['col_id'], cell_idx))

                            X_tmp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'og')].tolist())
                            y_tmp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'gt')].tolist())
                            original_data_values_tmp.append(
                                (row['table_id'], row['col_id'], cell_idx))

                    logger.info("Length of X_test: {}".format(len(X_test)))
                    logger.info("Length of X_tmp: {}".format(len(X_tmp)))

                    cells_per_cluster, labels_per_cluster, samples = sampling_labeling(X_tmp, y_tmp,
                                                                        n_cell_clusters_per_col_cluster_dict[c_idx], cell_clustering_alg)
                    labels += [original_data_values_tmp[sample] for sample in samples]
                    X_train, y_train = label_propagation(X_train, X_tmp, y_train, cells_per_cluster, labels_per_cluster)

                except Exception as e:
                    logger.error(e)

    with open(os.path.join(output_path, "original_data_values.pkl"), "wb") as filehandler:
        pickle.dump(original_data_values, filehandler)

    with open(os.path.join(results_path, "sampled_tuples.pkl"), "wb") as filehandler:
        pickle.dump(labels, filehandler)
        logger.info("Number of Labeled Cells: {}".format(len(labels)))

    return X_train, y_train, X_test, y_test, original_data_values, len(labels)


def classify(x_train, y_train, x_test):
    logger.info("Classification")
    imp = SimpleImputer(strategy="most_frequent")
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf = make_pipeline(imp, gbc)
    clf.fit(np.asarray(x_train), np.asarray(y_train))
    predicted = clf.predict(x_test)

    return predicted


def dask_classifier(x_train, y_train, x_test):
    logger.info("Classification - Parallel")
    with LocalCluster() as cluster:
        with Client(cluster) as client:
            clf = xgb.dask.DaskXGBClassifier()
            clf.client = client  # assign the client
            X_d_train = da.from_array(x_train, chunks=(1000, len(x_train[0])))
            y_d_train = da.from_array(y_train, chunks=1000)
            clf.fit(X_d_train, y_d_train)
            X_d_test = da.from_array(x_test, chunks=(1000, len(x_test[0])))
            predicted = clf.predict(X_d_test)
            np_predicted = np.array(predicted)

    return np_predicted


def error_detector(col_groups_files_path, output_path, results_path, features_dict, n_labels, number_of_clusters, cell_clustering_alg, classification_mode):
    X_train, y_train, X_test, y_test, original_data_values, n_samples = get_train_test_sets(col_groups_files_path, output_path,
                                                                                 results_path,
                                                                                 features_dict, n_labels,
                                                                                 number_of_clusters, cell_clustering_alg)
    if classification_mode == "parallel":
        predicted = dask_classifier(X_train, y_train, X_test)
    else:
        predicted = classify(X_train, y_train, X_test)

    return y_test, predicted, original_data_values, n_samples
