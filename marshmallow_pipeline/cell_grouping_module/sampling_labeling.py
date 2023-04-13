

import logging
from statistics import mode
import math
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest


logger = logging.getLogger()

def get_n_cell_groups_noise_based(noise_dict, cluster_sizes, labeling_budget):
    #TODO: datatype noise_dict
    df = pd.DataFrame.from_dict(noise_dict).sort_values(by=['n_noise'], ascending=False)
    cluster_sizes_df = pd.DataFrame.from_dict(cluster_sizes)
    size_dict = {"table_cluster": [], "col_cluster": [], "n_cells": []}
    for tab_cl in cluster_sizes:
        for col_cl in cluster_sizes[tab_cl]:
            size_dict["table_cluster"].append(tab_cl)
            size_dict["col_cluster"].append(int(col_cl))
            size_dict["n_cells"].append(cluster_sizes_df[tab_cl][col_cl])

    df = pd.merge(df, pd.DataFrame.from_dict(size_dict), on = ['table_cluster', 'col_cluster'])

    df["noise_ratio"] = df["n_noise"] / df["n_cells"]
    df = df.sort_values(by=['noise_ratio'], ascending=False)
    df["n_cell_groups"] = df.apply(lambda x: min(x["n_cells"], 2), axis=1)
    used_labels = 2 * df.shape[0]
    if labeling_budget > used_labels:
        sum_ratio = df["noise_ratio"].sum()
        df["n_cell_groups"] = \
            df.apply(lambda x: x["n_cell_groups"] + math.floor(min(x["n_cells"] - x["n_cell_groups"], x["noise_ratio"]/sum_ratio * (labeling_budget - used_labels))), axis=1)

    return df

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