import collections
import hashlib
import json
import logging
import math
import os
import pickle
import random
import numpy as np
import pandas as pd
import scipy
from sklearn import manifold
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.pipeline import make_pipeline
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

logger = logging.getLogger()


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def get_cells_features(sandbox_path, output_path, table_char_set_dict, tables_dict):

    features_dict = dict()
    table_id = 0
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
                    dirty_df = pd.read_csv(path + "/dirty_clean.csv", sep=",", header="infer", encoding="utf-8", dtype=str,
                                        keep_default_na=False,
                                        low_memory=False)
                    dirty_df = dirty_df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
                    dirty_df = dirty_df.replace('', 'NULL')
                    
                    clean_df = pd.read_csv(path + "/" + "clean.csv", sep=",", header="infer", encoding="utf-8",
                                        dtype=str, keep_default_na=False, low_memory=False)
                    clean_df = clean_df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
                    clean_df = clean_df.replace('', 'NULL')

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
                    logger.info("table_id: {}".format(table_id))
                except Exception as e:
                    logger.error(e)
                finally:
                    table_id += 1


    with open(os.path.join(output_path, "features.pickle"), "wb") as filehandler:
        pickle.dump(features_dict, filehandler)
    return features_dict


def sampling_labeling(table_cluster, col_cluster, x, y, n_cell_clusters_per_col_cluster, cells_clustering_alg, value_temp):
    # visulaize_cell_group(x, y, str(table_cluster) + "_" + str(col_cluster))
    logger.info("sampling_labeling")
    clustering = None 

    if cells_clustering_alg == "km":
        logging.info("KMeans - n_cell_clusters_per_col_cluster: {}".format(n_cell_clusters_per_col_cluster))
        if table_cluster == '33' and col_cluster == 27:
            for u in x:
                print(len(u))
        if len(x) < n_cell_clusters_per_col_cluster + 1:
            n_cell_clusters_per_col_cluster = len(x) - 1 
        clustering = MiniBatchKMeans(n_clusters=n_cell_clusters_per_col_cluster + 1, random_state=0, reassignment_ratio=0, batch_size = 256 * 64).fit(x)
        logging.info("KMeans - n_cell_clusters_generated: {}".format(len(set(clustering.labels_))))
        labels = clustering.labels_
        
    elif cells_clustering_alg == "dbscan":
        logging.info("DBSCAN - n_cell_clusters_per_col_cluster: {}".format(n_cell_clusters_per_col_cluster))
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(x)
        logging.info("DBSCAN - n_cell_clusters_generated: {}".format(len(set(clustering.labels_))))
        labels = clustering.labels_

    elif cells_clustering_alg == "hac":
        logging.info("HAC - n_cell_clusters_per_col_cluster: {}".format(n_cell_clusters_per_col_cluster))
        clustering = AgglomerativeClustering(n_clusters = n_cell_clusters_per_col_cluster + 1).fit(x)
        logging.info("HAC - n_cell_clusters_generated: {}".format(len(set(clustering.labels_))))
        labels = clustering.labels_

    elif cells_clustering_alg == "sl":
        logging.info("SL - n_cell_clusters_per_col_cluster: {}".format(n_cell_clusters_per_col_cluster))
        clustering = scipy.cluster.hierarchy.linkage(x, method='single', metric='euclidean')
        labels = scipy.cluster.hierarchy.fcluster(clustering, n_cell_clusters_per_col_cluster + 1, criterion="maxclust")
        logging.info("SL - n_cell_clusters_generated: {}".format(len(set(labels))))
    
    elif cells_clustering_alg == "fastcluster":
        logging.info("fast - n_cell_clusters_per_col_cluster: {}".format(n_cell_clusters_per_col_cluster))
        clustering = fastcluster.linkage_vector(x, method="single", metric="euclidean")
        labels = scipy.cluster.hierarchy.fcluster(clustering, n_cell_clusters_per_col_cluster + 1, criterion="maxclust")
        logging.info("fast - n_cell_clusters_generated: {}".format(len(set(labels))))
        
    logging.info("cells per cluster")
    cells_per_cluster = dict()
    # TODO remove this
    errors_per_cluster = dict()
    labels_per_cluster_all = dict()
    labels_per_cluster = dict()

    for cell in enumerate(labels):
        if cell[1] in cells_per_cluster.keys():
            cells_per_cluster[cell[1]].append(cell[0])
            if y[cell[0]] == 1:
                errors_per_cluster[cell[1]] += 1
        else:
            cells_per_cluster[cell[1]] = [cell[0]]
            labels_per_cluster_all[cell[1]] = []
            errors_per_cluster[cell[1]] = y[cell[0]]

    # error_probability_dict = dict()
    # for cpc in cells_per_cluster.keys():
    #     error_probability_dict[cpc] = (errors_per_cluster[cpc], errors_per_cluster[cpc] / len(cells_per_cluster[cpc]), len(cells_per_cluster[cpc]))
    # logger.info("error_probability_dict: {}".format(error_probability_dict))
    # with open('/home/fatemeh/ED-Scale/EDS/logs/errors_prob/error_probability_dict_table_cluster_{}_col_cluster{}.json'.format(str(table_cluster), str(col_cluster)), 'wb') as fp:
    #     pickle.dump(error_probability_dict, fp)

    
    samples = []
    sample_values = []
    samples_orig_values = []
            
    if cells_clustering_alg == "km":
        cells_per_cluster = dict(sorted(cells_per_cluster.items(), key=lambda x: x[1]))
        for cluster in cells_per_cluster.keys():
            center = clustering.cluster_centers_[cluster]
            cluster_points = np.array([x[idx] for idx in cells_per_cluster[cluster]])
            distances = np.linalg.norm(cluster_points - np.array([center]), axis=1)
            closest = np.argsort(distances)
            while len(samples) < n_cell_clusters_per_col_cluster and len(closest) > 0:
                sample_idx = cells_per_cluster[cluster][closest[0]]
                if x[sample_idx] not in sample_values:
                    samples.append(sample_idx)
                    sample_values.append(x[sample_idx])
                    samples_orig_values.append(value_temp[sample_idx])
                closest = closest[1:]
    else:
        logging.info("sampling_labeling - else")
        cells_per_cluster = collections.OrderedDict(cells_per_cluster)
        for cluster in cells_per_cluster.keys():
            max_iter = 100
            while len(samples) < n_cell_clusters_per_col_cluster and max_iter > 0:
                sample_idx = cells_per_cluster[cluster][np.random.randint(0, len(cells_per_cluster[cluster]))]
                logging.info("sample_idx: {}".format(sample_idx))
                if x[sample_idx] not in sample_values:
                    samples.append(sample_idx)
                    sample_values.append(x[sample_idx])
                    samples_orig_values.append(value_temp[sample_idx])
                else:
                    max_iter -= 1

    logger.info("labeling")

    for cell in enumerate(labels):
        if cell[0] in samples:
            labels_per_cluster_all[cell[1]].append(y[cell[0]])
    
    for c in labels_per_cluster_all.keys():
        if len(labels_per_cluster_all[c]) > 0:
            labels_per_cluster[c] = mode(labels_per_cluster_all[c])  

    universal_samples = dict()
    for s in samples:
        universal_samples[(table_cluster, col_cluster, s)] = y[s]

    logger.info("labels_per_cluster: {}".format(labels_per_cluster))
    logger.info("samples: {}".format(samples_orig_values))
    # visualize_cell_clusters(x, y, cells_per_cluster, samples)
    return cells_per_cluster, labels_per_cluster, universal_samples, samples

# def visualize_cell_clusters(x, y, cells_per_cluster, samples):
#     logger.info("visualize_cell_clusters")
#     pca = PCA(n_components=2, svd_solver='full')
#     logger.info("pca fit_transform")
#     # fit and transform
#     mnist_tr = pca.fit_transform(x)
#     cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
#                        data=np.column_stack((mnist_tr, y)))

#     cps_df.loc[:, 'target'] = cps_df.target.astype(int)
#     d = {0: 'correct', 1: 'error'}
#     cps_df.loc[:, 'target'] = cps_df.target.map(d)

#     fig, ax = plt.subplots()

#     for cell_cluster in cells_per_cluster.keys():
#         cell_cluster_name = "cell_cluster_{}".format(cell_cluster)
#         logger.info("cell_cluster_name: {}".format(cell_cluster_name))
#         points = get_points_of_cell_cluster(mnist_tr, cells_per_cluster, cell_cluster)
#         plt.scatter(points[:,0] , points[:,1] , color = 'red')
#         ax.scatter(points[:, 0], points[:, 1], edgecolors='black', facecolors='none', s=200, label=f'Cluster {cell_cluster}')
#         # ax.scatter(points[i, 0], points[i, 1], marker='o', c='red', s=100, edgecolors='black')
    
#     ax.legend()
#     plt.show()


    # grid = sns.FacetGrid(cps_df, hue="target")
    # fig = grid.map(plt.scatter, 'CP1', 'CP2').add_legend()
    # fig.savefig(os.path.join('/home/fatemeh/ED-Scale/logs/cell_clusters/{}.png'.format(cell_cluster_name)))
    return 

def get_points_of_cell_cluster(x, cells_per_cluster, cell_cluster):
    logger.info("get_points_of_cell_cluster {}".format(cell_cluster))
    points = []
    for cell in cells_per_cluster[cell_cluster]:
        points.append(x[cell])
    return points

def visulaize_cell_group(x, y, name):
    #logger.info("x: {}".format(x))
    n_features = len(x[0])
    logger.info("n_features: {}".format(n_features))
    logger.info("visualize_cell_group name {}".format(name))
    logger.info("converting x to numpy array")
    x = np.array(x)
    logger.info("x[0]: {}".format(x[0]))
    logger.info("x.shape: {}".format(x.shape))
    logger.info("converting y to numpy array")
    y = np.array(y)

    logger.info("pca")
    # dimensionality reduction using t-SNE
    pca = PCA(n_components=2, svd_solver='full')

    logger.info("pca fit_transform")
    # fit and transform
    mnist_tr = pca.fit_transform(x)
    logger.info("mnist_tr.shape: {}".format(mnist_tr.shape))
    cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
                       data=np.column_stack((mnist_tr, y)))

    cps_df.loc[:, 'target'] = cps_df.target.astype(int)
    d = {0: 'correct', 1: 'error'}
    cps_df.loc[:, 'target'] = cps_df.target.map(d)
    grid = sns.FacetGrid(cps_df, hue="target")
    fig = grid.map(plt.scatter, 'CP1', 'CP2').add_legend()
    fig.savefig(os.path.join('/home/fatemeh/ED-Scale/outputs/complete-0802/check_col_groups_without_header/{}.png'.format(name)))
    return

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


def get_number_of_clusters(col_groups_dir):
    number_of_col_clusters = 0
    for file in os.listdir(col_groups_dir):
        if ".pickle" in file:
            with open(os.path.join(col_groups_dir, file), 'rb') as filehandler:
                group_df = pickle.load(filehandler)
                number_of_clusters = len(group_df['column_cluster_label'].unique())
                number_of_col_clusters += number_of_clusters
    return number_of_col_clusters

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
    while assigned_labels < init_n_labels:
        n_labels -= assigned_labels
        n_cell_clusters_per_col_cluster = math.floor(n_labels / number_of_all_col_clusters)
        if n_cell_clusters_per_col_cluster >= 1:
            for table_cluster in cluster_sizes.keys():
                for col_cluster in cluster_sizes[table_cluster].keys():
                    current_labels= n_cell_clusters_per_col_cluster_dict[(table_cluster, col_cluster)]
                    if current_labels  + n_cell_clusters_per_col_cluster + 1 < cluster_sizes[table_cluster][col_cluster]:
                        n_assigned_labels = current_labels + n_cell_clusters_per_col_cluster
                    else:
                        n_assigned_labels = cluster_sizes[table_cluster][col_cluster]
                    n_cell_clusters_per_col_cluster_dict[(table_cluster, col_cluster)] = n_assigned_labels
                    assigned_labels += n_assigned_labels
        else:
            while assigned_labels < init_n_labels:
                rand = random.choice(list(n_cell_clusters_per_col_cluster_dict.keys()))
                if n_cell_clusters_per_col_cluster_dict[rand] + 1 < cluster_sizes[rand[0]][rand[1]]:
                    n_cell_clusters_per_col_cluster_dict[rand] += 1
                    assigned_labels += 1
    return n_cell_clusters_per_col_cluster_dict

def process_col_cluster(n_cell_clusters_per_col_cluster, table_cluster, cluster,\
                         group_df, features_dict, cell_clustering_alg):
    X_train = []
    y_train = []
    X_temp = []
    y_temp = []
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
                datacells_uids[(row['table_id'], row['col_id'], cell_idx, row['col_value'][cell_idx])] = current_local_cell_uid
                current_local_cell_uid += 1

        cells_per_cluster, labels_per_cluster, universal_samples, samples = sampling_labeling(table_cluster, cluster, X_temp, y_temp,
                                                            n_cell_clusters_per_col_cluster, cell_clustering_alg, value_temp)
        X_labeled_by_user.extend([X_temp[sample] for sample in samples])
        y_labeled_by_user.extend([y_temp[sample] for sample in samples])

        X_train, y_train, X_test, y_test, y_cell_ids = \
                get_train_test_sets(X_temp, y_temp, samples, cells_per_cluster, labels_per_cluster)
        predicted = classify(X_train, y_train, X_test)

    except Exception as e:
        logger.error(e)
        logging.error("error", e)

    return y_test, y_cell_ids, predicted, original_data_keys_temp, universal_samples, X_labeled_by_user, y_labeled_by_user, datacells_uids

def generate_bigrams(chars):
    bigrams = set()
    for i in range(len(chars)):
        for j in range(i+1,len(chars)):
            bigrams.add(chars[i] + chars[j])
            bigrams.add(chars[j] + chars[i])
    return bigrams

def error_detector(cell_feature_generator_enabled, sandbox_path, col_groups_dir, output_path, results_path, n_labels, number_of_col_clusters, cluster_sizes, cell_clustering_alg, tables_dict):

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

    n_cell_clusters_per_col_cluster_dict = get_n_cell_clusters_per_col_cluster_dict(n_labels, cluster_sizes, number_of_col_clusters)
    logging.info("n_cell_clusters_per_col_cluster_dict: {}".format(n_cell_clusters_per_col_cluster_dict))

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
                # charset_val = [ch for v in group_df[group_df['column_cluster_label'] == cluster]["col_chars"].values for ch in v]
                # bigrams = generate_bigrams(charset_val)
                charset_val = set(charset_val)
                # charset_val.update(bigrams)
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
    # TODO
        # with open(os.path.join(output_path, configs["DIRECTORIES"]["cell_features_filename"]), 'rb') as file:
        #     features_dict = pickle.load(file)
        #     logger.info("Cell features loaded.")

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
                y_test, y_cell_ids, predicted, original_data_keys_temp, universal_samples, \
                X_labeled_by_user, y_labeled_by_user, datacells_local_ids = \
                     process_col_cluster(n_cell_clusters_per_col_cluster_dict[(str(table_cluster), str(cluster))], table_cluster, cluster, group_df, features_dict, cell_clustering_alg)
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

