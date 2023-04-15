

import logging
from statistics import mode
import math
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest


logger = logging.getLogger()

def join_noise_dict_clusters(noise_dict, cluster_sizes):
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
    return df

def get_n_cell_groups_noise_based(noise_dict, cluster_sizes, labeling_budget):
    df = join_noise_dict_clusters(noise_dict, cluster_sizes)
    df["n_cell_groups"] = df.apply(lambda x: min(2, x["n_cells"]), axis = 1)
    df["sampled"] = df.apply(lambda x: False, axis=1)
    used_labels = df["n_cell_groups"].sum()
    if labeling_budget > used_labels:
        sum_ratio = df["noise_ratio"].sum()
        df["n_cell_groups"] = \
            df.apply(lambda x: x["n_cell_groups"] + math.floor(min(x["n_cells"] - x["n_cell_groups"], x["noise_ratio"]/sum_ratio * (labeling_budget - used_labels))), axis=1)
    while labeling_budget > df["n_cell_groups"].sum():
        df = df.sort_values(by=['n_cell_groups'], ascending=True)
        df["n_cell_groups"].iloc[0] = df["n_cell_groups"].iloc[0] + 1
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

    clustering = None 
    logger.info("cells per cluster")
    cells_per_cluster = dict()
    # TODO remove this
    errors_per_cluster = dict()
    labels_per_cluster_all = dict()
    labels_per_cluster = dict()
    universal_samples = dict()
    samples = []
    sample_values = []
    samples_orig_values = []

    if cells_clustering_alg == "km":
        if len(outliers) != 0:
            n_cell_clusters_per_col_cluster = min(len(outliers),n_cell_clusters_per_col_cluster)

            logger.info("Clustering Outliers")
            logger.info("KMeans - n_cell_clusters_per_col_cluster: {}".format(n_cell_clusters_per_col_cluster))
            if n_cell_clusters_per_col_cluster <= 0:
                print("******************")
            clustering = MiniBatchKMeans(n_clusters= int(n_cell_clusters_per_col_cluster), random_state=0, reassignment_ratio=0, batch_size = 256 * 64).fit(outliers)
            logger.info("KMeans - n_cell_clusters_generated: {}".format(len(set(clustering.labels_))))
            clustering_labels = clustering.labels_
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

            cells_per_cluster[len(cells_per_cluster.keys())] = non_outliers
            if cells_clustering_alg == "km":
                for cluster in list(cells_per_cluster.keys())[:-1]:
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
            
            for s in samples:
                universal_samples[(table_cluster, col_cluster, s)] = y[s]
                
        else:
            
            n_cell_clusters_per_col_cluster = min(len(non_outliers),n_cell_clusters_per_col_cluster)
            
            logger.info("Clustering Non Outliers")
            logger.info("KMeans - n_cell_clusters_per_col_cluster: {}".format(n_cell_clusters_per_col_cluster))
            clustering = MiniBatchKMeans(n_clusters= int(n_cell_clusters_per_col_cluster), random_state=0, reassignment_ratio=0, batch_size = 256 * 64).fit(non_outliers_features)
            logger.info("KMeans - n_cell_clusters_generated: {}".format(len(set(clustering.labels_))))
            clustering_labels = clustering.labels_
            for cell in enumerate(clustering_labels):
                orig_x_idx = non_outliers[cell[0]]
                if cell[1] in cells_per_cluster.keys():
                    cells_per_cluster[cell[1]].append(orig_x_idx)
                    if y[orig_x_idx] == 1:
                        errors_per_cluster[cell[1]] += 1
                else:
                    cells_per_cluster[cell[1]] = [orig_x_idx]
                    labels_per_cluster_all[cell[1]] = []
                    errors_per_cluster[cell[1]] = y[orig_x_idx]
            cells_per_cluster[len(cells_per_cluster.keys())] = outliers
            if cells_clustering_alg == "km":
                for cluster in list(cells_per_cluster.keys())[:-1]:
                    points = [x[i] for i in cells_per_cluster[cluster]]
                    center_point = np.array(center_point)
                    distances = np.linalg.norm(points - center_point, axis=1)
                    samples_tmp_nout_idx = np.argmin(distances)
                    samples_tmp = [cells_per_cluster[cluster][samples_tmp_nout_idx]]
                    samples.extend(samples_tmp) # sample k vectors
                    sample_values.extend([x[i] for i in samples_tmp]) # get the selected vectors
                    samples_orig_values.extend([value_temp[i] for i in samples_tmp]) # get the selected vectors values

            logger.info("labeling")
            for cell in enumerate(clustering_labels):
                orig_x_idx = non_outliers[cell[0]]
                if orig_x_idx in samples:
                    labels_per_cluster_all[cell[1]].append(y[orig_x_idx])
            
            for c in labels_per_cluster_all.keys():
                if len(labels_per_cluster_all[c]) > 0:
                    labels_per_cluster[c] = mode(labels_per_cluster_all[c])  
            
            for s in samples:
                universal_samples[(table_cluster, col_cluster, s)] = y[s]

    remained_labels = n_cell_clusters_per_col_cluster - len(samples)
    return cells_per_cluster, labels_per_cluster, universal_samples, samples, remained_labels