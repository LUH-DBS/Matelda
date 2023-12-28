import copy
import logging
import math
import random
from statistics import mode
from sklearn.feature_selection import VarianceThreshold

from sklearn.neighbors import NearestNeighbors
from marshmallow_pipeline.cell_grouping_module.llm_labeling import few_shot_prediction, get_foundation_model_prediction

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import MiniBatchKMeans


def get_n_labels(cluster_sizes_df, labeling_budget, min_num_labes_per_col_cluster):
    cluster_sizes_df["n_labels"] = cluster_sizes_df.apply(
        lambda x: min(min_num_labes_per_col_cluster, x["n_cells"]), axis=1
    )
    cluster_sizes_df["sampled"] = cluster_sizes_df.apply(lambda x: False, axis=1)
    used_labels = cluster_sizes_df["n_labels"].sum()
    num_total_cells = cluster_sizes_df["n_cells"].sum()
    if labeling_budget > used_labels:
        remaining_labels = labeling_budget - used_labels
        cluster_sizes_df["n_labels"] = cluster_sizes_df.apply(
            lambda x: x["n_labels"]
            + math.floor(
                min(
                    x["n_cells"] - x["n_labels"],
                    (x["n_cells"] / num_total_cells) * remaining_labels,
                )
            ),
            axis=1,
        )
    i = 0
    j = 0
    cluster_sizes_df.sort_values(by=["n_cells"], ascending=False, inplace=True)
    while labeling_budget > cluster_sizes_df["n_labels"].sum() and j < len(cluster_sizes_df["n_cells"]):
        if cluster_sizes_df["n_labels"].iloc[i] < cluster_sizes_df["n_cells"].iloc[i]:
            cluster_sizes_df["n_labels"].iloc[i] = \
                cluster_sizes_df["n_labels"].iloc[i] + 1
            j = 0
        else:
            j += 1
        if i < len(cluster_sizes_df["n_cells"]) - 1:
            i += 1
        else:
            i = 0
    return cluster_sizes_df


def cell_clustering(table_cluster, col_cluster, x, y, n_cell_clusters_per_col_cluster, n_cores, labels_per_cell_group):
    # logging.info(
    #     "Cell Clustering - table_cluster: %s, col_cluster: %s",
    #     table_cluster,
    #     col_cluster,
    # )
    clustering = None
    cells_per_cluster = {}
    errors_per_cluster = {}
    cell_clustering_dict = {
        "table_cluster": [],
        "col_cluster": [],
        "n_cells": [],
        "n_init_labels": [],
        "n_produced_cell_clusters": [],
        "n_current_requiered_labels": [],
        "remaining_labels": [],
        "cells_per_cluster": [],
        "errors_per_cluster": [],
    }
    n_cell_clusters_per_col_cluster = min(len(x), n_cell_clusters_per_col_cluster)
    logging.debug(
        "KMeans - n_cell_clusters_per_col_cluster: %s", n_cell_clusters_per_col_cluster
    )
    clustering = MiniBatchKMeans(
        n_clusters=int(n_cell_clusters_per_col_cluster),
        batch_size=256 * n_cores,
    ).fit(x)
    set_clustering_labels = set(clustering.labels_)
    logging.debug("KMeans - n_cell_clusters_generated: %s", len(set_clustering_labels))
    clustering_labels = clustering.labels_
    for cell in enumerate(clustering_labels):
        if cell[1] in cells_per_cluster:
            cells_per_cluster[cell[1]].append(cell[0])
            if y[cell[0]] == 1:
                errors_per_cluster[cell[1]] += 1
        else:
            cells_per_cluster[cell[1]] = [cell[0]]
            errors_per_cluster[cell[1]] = y[cell[0]]

    cell_clustering_dict["table_cluster"] = table_cluster
    cell_clustering_dict["col_cluster"] = col_cluster
    cell_clustering_dict["n_cells"] = len(x)
    cell_clustering_dict["n_init_labels"] = n_cell_clusters_per_col_cluster * labels_per_cell_group
    cell_clustering_dict["n_produced_cell_clusters"] = len(set_clustering_labels)
    cell_clustering_dict["n_current_requiered_labels"] = len(set_clustering_labels) * labels_per_cell_group
    
    cell_clustering_dict["remaining_labels"] = (
        cell_clustering_dict["n_init_labels"]
        - cell_clustering_dict["n_current_requiered_labels"]
    )
    cell_clustering_dict["cells_per_cluster"] = cells_per_cluster
    cell_clustering_dict["errors_per_cluster"] = errors_per_cluster

    return cell_clustering_dict


def update_n_labels(cell_clustering_recs):
    logging.info("Update n_labels")
    cell_clustering_df = pd.DataFrame(cell_clustering_recs)
    remaining_labels = cell_clustering_df["remaining_labels"].sum()
    cell_clustering_df["n_labels_updated"] = cell_clustering_df[
        "n_current_requiered_labels"
    ]
    if remaining_labels == 0:
        return cell_clustering_df

    elif remaining_labels > 0:
        cell_clustering_df.sort_values(
            by=["n_produced_cell_clusters"], ascending=False, inplace=True
        )
        i = 0
        j = 0
        while remaining_labels > 0 and j < len(cell_clustering_df):
            if (
                cell_clustering_df["n_cells"].iloc[i]
                > cell_clustering_df["n_labels_updated"].iloc[i]
            ):
                cell_clustering_df["n_labels_updated"].iloc[i] += 1
                remaining_labels -= 1
                j = 0
            else:
                j += 1     
            if i < len(cell_clustering_df) - 1:
                i += 1
            else:
                i = 0
    elif remaining_labels < 0:
        logging.info("remaining_labels < 0 - remaining_labels: {}".format(remaining_labels))
        logging.info("I need more labels :)")
    logging.info("Update n_labels - remaining_labels: {}".format(remaining_labels))
    return cell_clustering_df


def get_the_nearest_point_to_centroid(feature_vectors):
    centroid = np.mean(feature_vectors, axis=0)
    closest_index = min(range(len(feature_vectors)), key=lambda i: euclidean(feature_vectors[i], centroid))
    return closest_index


def split_cell_cluster(cell_cluster_n_labels, n_cores, x_cluster, y_cluster, col_group_cell_idx, updated_cells_per_cluster, updated_errors_per_cluster, updated_cell_cluster_n_labels, cluster, min_n_labels_per_cell_group):
    try:
        clustering = MiniBatchKMeans(n_clusters=min(len(x_cluster), math.floor(cell_cluster_n_labels[cluster]/min_n_labels_per_cell_group)), batch_size=256 * n_cores).fit(x_cluster)
        set_clustering_labels = set(clustering.labels_)
        logging.info("inner cluster splitting - n_clusters: %s", len(set_clustering_labels))
        clustering_labels = clustering.labels_
        if len(set_clustering_labels) < cell_cluster_n_labels[cluster]:
            return updated_cells_per_cluster, updated_errors_per_cluster, updated_cell_cluster_n_labels
        else:
            x_cluster_splited = []
            y_cluster_splited = []
            x_idx_cluster_splited = []
            for i in set_clustering_labels:
                x_cluster_splited.append([])
                y_cluster_splited.append([])
                x_idx_cluster_splited.append([])
            for x_idx, x in enumerate(x_cluster):
                x_cluster_splited[clustering_labels[x_idx]].append(x)
                x_idx_cluster_splited[clustering_labels[x_idx]].append(x_idx)
                y_cluster_splited[clustering_labels[x_idx]].append(y_cluster[x_idx])

            for mini_cluster in range(len(x_cluster_splited)):
                updated_cells_per_cluster[max(updated_cells_per_cluster.keys())+1] = [col_group_cell_idx[x_idx] for x_idx in x_idx_cluster_splited[mini_cluster]]
                updated_errors_per_cluster[max(updated_errors_per_cluster.keys())+1] = sum(y_cluster_splited[mini_cluster])
                updated_cell_cluster_n_labels[max(updated_cell_cluster_n_labels.keys())+1] = 1
            updated_cells_per_cluster.pop(cluster)
            updated_errors_per_cluster.pop(cluster)
            updated_cell_cluster_n_labels.pop(cluster)
    except Exception as e:
        logging.error("inner cluster splitting - error: %s", e)
    return updated_cells_per_cluster, updated_errors_per_cluster, updated_cell_cluster_n_labels

def distribute_labels_in_cell_clusters(cell_cluster_n_labels, sorted_clusters, values_per_cluster, n_labels, min_n_labels_per_cell_group=1):
    i = 0
    sorted_cluster_idx = 0
    if min_n_labels_per_cell_group == 2:
        while sorted_cluster_idx < len(sorted_clusters) and n_labels > 1:
            cluster = sorted_clusters[sorted_cluster_idx]
            set_values_per_cluster = set(values_per_cluster[cluster])
            # Compare the number of labels with the number of unique feature vectors
            if cell_cluster_n_labels[cluster] < len(set_values_per_cluster):
                cell_cluster_n_labels[cluster] += 2
                n_labels -= 2
            if cell_cluster_n_labels[cluster] == 0:
                print("I am here")
            sorted_cluster_idx += 1

    i = 0
    sorted_cluster_idx = 0
    while n_labels > 0 and i < len(sorted_clusters):
        cluster = sorted_clusters[sorted_cluster_idx]
        set_values_per_cluster = set(values_per_cluster[cluster])
        # Compare the number of labels with the number of unique feature vectors
        if cell_cluster_n_labels[cluster] < len(set_values_per_cluster):
            cell_cluster_n_labels[cluster] += 1
            n_labels -= 1
            i = 0
        else:
            i += 1
        if sorted_cluster_idx < len(sorted_clusters) - 1:
            sorted_cluster_idx += 1
        else:
            sorted_cluster_idx = 0
    return cell_cluster_n_labels
    
def pick_samples_in_cell_cluster(cluster, updated_cells_per_cluster, updated_cell_cluster_n_labels, 
                                 x, y, dirty_cell_values, tables_tuples_dict, nearest_neighbours_percentage, semi_propagation, labeling_method, llm_labels_per_cell_group, original_data_keys_temp, min_n_labels_per_cell_group):
    try:
        x_cluster = []
        y_cluster = []
        key_cluster = []
        samples_feature_vectors = []
        samples_labels = []
        samples_indices_global = []
        samples_indices_cell_group = []
        dirty_cell_values_cluster = []
        n_user_labeled_cells = 0
        n_model_labeled_cells = 0
        cells_with_propagated_labels_vectors = []
        cells_with_propagated_labels_cg_idx = []
        cells_with_propagated_labels_gl_idx = []
        y_cells_with_propagated_labels = []
        col_group_cell_idx = updated_cells_per_cluster[cluster]
        for cell_idx in col_group_cell_idx:
            x_cluster.append(x[cell_idx])
            y_cluster.append(y[cell_idx])
            key_cluster.append(original_data_keys_temp[cell_idx])

        if updated_cell_cluster_n_labels[cluster] > 1:
            sampled_cells = []
            samples_idx = []
            samples_labels = []
            user_samples = []
            outer_while_trial = 5
            while len(samples_feature_vectors) < updated_cell_cluster_n_labels[cluster] and outer_while_trial > 0:
                trial = 5
                unique_sample = True
                sample = np.random.randint(0, len(x_cluster))
                while trial > 0 and sample in user_samples:
                    sample = np.random.randint(0, len(x_cluster))
                    trial -= 1
                if trial == 0 and sample in user_samples:
                    unique_sample = False
                if unique_sample:
                    if semi_propagation:
                        knn = NearestNeighbors(n_neighbors= round(nearest_neighbours_percentage * len(x_cluster)), metric="euclidean")
                        knn.fit(x_cluster)
                        distances, indices = knn.kneighbors([x_cluster[sample]])
                        for idx in indices[0]:
                            cells_with_propagated_labels_cg_idx.append(idx)
                            cells_with_propagated_labels_gl_idx.append(col_group_cell_idx[idx])
                            cells_with_propagated_labels_vectors.append(x_cluster[idx])
                            y_cells_with_propagated_labels.append(y_cluster[idx])
                    user_samples.append(sample)
                    samples_feature_vectors.append(x_cluster[sample])
                    if labeling_method == 0:
                        samples_labels.append(y_cluster[sample])
                        n_user_labeled_cells += 1
                    elif labeling_method == 1:
                        label = get_foundation_model_prediction(tables_tuples_dict, key_cluster[sample])
                        samples_labels.append(label) 
                        n_model_labeled_cells += 1
                    elif labeling_method == 2:
                        sampled_cells.append(key_cluster[sample])
                        samples_idx.append(sample)
                        samples_labels.append(y_cluster[sample])
                    dirty_cell_values_cluster.append(
                        dirty_cell_values[col_group_cell_idx[sample]]
                    )
                    if col_group_cell_idx[sample] in samples_indices_global:
                        logging.INFO("sample is already in samples_indices_global")
                    samples_indices_global.append(col_group_cell_idx[sample])
                    samples_indices_cell_group.append(sample)
                else: 
                    outer_while_trial -= 1
            if labeling_method == 2:
                n_samples_llm = min(llm_labels_per_cell_group, len(x_cluster) - min_n_labels_per_cell_group)
                llm_samples_idx = get_random_sample_list(n_samples_llm, x_cluster, user_samples)
                llm_samples = [key_cluster[i] for i in llm_samples_idx]
                if len(llm_samples) > 0:
                    user_samples_labels_dict = {}
                    for samp_idx, sample in enumerate(sampled_cells):
                        user_samples_labels_dict[sample] = samples_labels[samp_idx]
                    test_labels = few_shot_prediction(tables_tuples_dict, llm_samples, user_samples_labels_dict) 
                    n_user_labeled_cells += len(sampled_cells)
                    n_model_labeled_cells += len(test_labels)
                    
                    for s_llm_idx, s_llm in enumerate(llm_samples_idx):
                        dirty_cell_values_cluster.append(
                            dirty_cell_values[col_group_cell_idx[s_llm]]
                        )
                        samples_feature_vectors.append(x_cluster[s_llm])
                        if col_group_cell_idx[s_llm] in samples_indices_global:
                            logging.INFO("sample is already in samples_indices_global")
                        samples_indices_global.append(col_group_cell_idx[s_llm])
                        samples_indices_cell_group.append(s_llm)
                        samples_labels.append(test_labels[s_llm_idx])
                
        else:
            sample = get_the_nearest_point_to_centroid(x_cluster)
            samples_feature_vectors.append(x_cluster[sample])
            if labeling_method == 0:
                samples_labels.append(y_cluster[sample])
                n_user_labeled_cells += 1
            elif labeling_method == 1:
                label = get_foundation_model_prediction(tables_tuples_dict, key_cluster[sample])
                samples_labels.append(label)
                n_model_labeled_cells += 1
            dirty_cell_values_cluster.append(
                dirty_cell_values[col_group_cell_idx[sample]]
            )
            if col_group_cell_idx[sample] in samples_indices_global:
                logging.INFO("sample is already in samples_indices_global")
            samples_indices_global.append(col_group_cell_idx[sample])
            samples_indices_cell_group.append(sample)

            if semi_propagation:
                knn = NearestNeighbors(n_neighbors= round(nearest_neighbours_percentage * len(x_cluster)), metric="euclidean")
                knn.fit(x_cluster)
                distances, indices = knn.kneighbors([x_cluster[sample]])
                for idx in indices[0]:
                    cells_with_propagated_labels_cg_idx.append(idx)
                    cells_with_propagated_labels_gl_idx.append(col_group_cell_idx[idx])
                    cells_with_propagated_labels_vectors.append(x_cluster[idx])
                    y_cells_with_propagated_labels.append(y_cluster[idx])
    except Exception as e:
        logging.error("pick_samples_in_cell_cluster - error: %s", e)
    return samples_feature_vectors, samples_labels, samples_indices_global, samples_indices_cell_group,\
          dirty_cell_values_cluster, n_user_labeled_cells, n_model_labeled_cells, \
        cells_with_propagated_labels_vectors, cells_with_propagated_labels_cg_idx, cells_with_propagated_labels_gl_idx, y_cells_with_propagated_labels

def get_random_sample_list(n_samples, x_cluster, user_samples):
    llm_samples_idx = []
    for i in range(n_samples):
        trial = 5
        sample_llm = np.random.randint(0, len(x_cluster))
        while (sample_llm in llm_samples_idx or sample_llm in user_samples) and trial > 0:
            sample_llm = np.random.randint(0, len(x_cluster))
            trial -= 1
        if trial == 0 and sample_llm in llm_samples_idx:
            continue
        else:
            llm_samples_idx.append(sample_llm)
    return llm_samples_idx

def check_and_split_cell_clusters(x, y, labeled_clusters, cell_cluster_n_labels, cells_per_cluster, n_cores, updated_cells_per_cluster, updated_errors_per_cluster, updated_cell_cluster_n_labels, min_n_labels_per_cell_group):
    for cluster in labeled_clusters:
        if cell_cluster_n_labels[cluster] > min_n_labels_per_cell_group:
            x_cluster = []
            y_cluster = []
            col_group_cell_idx = cells_per_cluster[cluster]
            for cell_idx in col_group_cell_idx:
                x_cluster.append(x[cell_idx])
                y_cluster.append(y[cell_idx])
            updated_cells_per_cluster, updated_errors_per_cluster, updated_cell_cluster_n_labels = \
                split_cell_cluster(cell_cluster_n_labels, 
                                   n_cores, 
                                   x_cluster, 
                                   y_cluster, 
                                   col_group_cell_idx, 
                                   updated_cells_per_cluster, 
                                   updated_errors_per_cluster,
                                   updated_cell_cluster_n_labels, 
                                   cluster, min_n_labels_per_cell_group)
    return updated_cells_per_cluster, updated_errors_per_cluster, updated_cell_cluster_n_labels

def update_samples_dict(cell_clustering_dict, samples_dict, cluster, \
                        samples_feature_vectors, samples_labels, samples_indices_global, \
                            samples_indices_cell_group, dirty_cell_values_cluster, \
                                updated_cells_per_cluster, updated_errors_per_cluster, updated_cell_cluster_n_labels,\
                                    cells_with_propagated_labels_vectors, cells_with_propagated_labels_cg_idx, cells_with_propagated_labels_gl_idx, y_cells_with_propagated_labels):
    logging.debug("Update samples dict")
    cell_clustering_dict["cells_per_cluster"].values[0] = updated_cells_per_cluster
    cell_clustering_dict["errors_per_cluster"].values[0] = updated_errors_per_cluster
    cell_clustering_dict["n_labels_updated"].values[0] = sum(updated_cell_cluster_n_labels.values())
    samples_dict["cell_cluster"].append(cluster)
    samples_dict["samples"].append(samples_feature_vectors)
    samples_dict["n_samples"].append(len(samples_feature_vectors))
    samples_dict["labels"].append(samples_labels)
    samples_dict["dirty_cell_values"].append(dirty_cell_values_cluster)
    samples_dict["samples_indices_cell_group"].append(samples_indices_cell_group)
    samples_dict["samples_indices_global"].append(samples_indices_global)
    samples_dict["cells_with_propagated_labels_vectors"].append(cells_with_propagated_labels_vectors)
    samples_dict["cells_with_propagated_labels_cg_idx"].append(cells_with_propagated_labels_cg_idx)
    samples_dict["cells_with_propagated_labels_gl_idx"].append(cells_with_propagated_labels_gl_idx)
    samples_dict["y_cells_with_propagated_labels"].append(y_cells_with_propagated_labels)
    return cell_clustering_dict, samples_dict


def sampling(cell_clustering_dict, x, y, dirty_cell_values, original_data_keys_temp, n_cores, tables_tuples_dict, nearest_neighbours_percentage, semi_propagation, labeling_method, llm_labels_per_cell_group, min_n_labels_per_cell_group):
    logging.debug("Sampling")
    samples_dict = {
        "cell_cluster": [],
        "samples": [],
        "n_samples": [],
        "samples_indices_cell_group": [],
        "samples_indices_global": [],
        "labels": [],
        "dirty_cell_values": [],
        "cells_with_propagated_labels_vectors": [],
        "cells_with_propagated_labels_cg_idx": [],
        "cells_with_propagated_labels_gl_idx": [],
        "y_cells_with_propagated_labels": []
    }

    cells_per_cluster = cell_clustering_dict["cells_per_cluster"].values[0]
    updated_cells_per_cluster = copy.deepcopy(cell_clustering_dict["cells_per_cluster"].values[0])
    updated_errors_per_cluster = copy.deepcopy(cell_clustering_dict["errors_per_cluster"].values[0])
    labeled_clusters = {
        key: value
        for key, value in cells_per_cluster.items()
    }
    sorted_clusters = sorted(labeled_clusters, key=lambda k: len(labeled_clusters[k]), reverse=True)
    logging.debug("Sampling - sorted_clusters: {}".format(sorted_clusters))
    cell_cluster_n_labels = {k: 0 for k in cells_per_cluster.keys()}
    updated_cell_cluster_n_labels = {k: 0 for k in cells_per_cluster.keys()}
    values_per_cluster = {k: [] for k in cells_per_cluster.keys()}
    for k in values_per_cluster.keys():
        values_per_cluster[k] = [tuple(x[i]) for i in cells_per_cluster[k]]
    n_labels = cell_clustering_dict["n_labels_updated"].values[0]        
    cell_cluster_n_labels = distribute_labels_in_cell_clusters(cell_cluster_n_labels, sorted_clusters, values_per_cluster, n_labels, min_n_labels_per_cell_group)
    updated_cell_cluster_n_labels = copy.deepcopy(cell_cluster_n_labels)

    updated_cells_per_cluster, updated_errors_per_cluster, updated_cell_cluster_n_labels = \
    check_and_split_cell_clusters(x, y, labeled_clusters, cell_cluster_n_labels, cells_per_cluster, n_cores, updated_cells_per_cluster, updated_errors_per_cluster, updated_cell_cluster_n_labels, min_n_labels_per_cell_group)
            
    updated_labeled_clusters = {
        key: value
        for key, value in updated_cells_per_cluster.items()
    }

    logging.debug("Number of updated_labeled_clusters: %s", len(updated_labeled_clusters))
    global_n_user_labeled_cells = 0
    global_n_model_labeled_cells = 0
    for cluster in updated_labeled_clusters:
        logging.debug("Sampling - cluster: %s", cluster)
        samples_feature_vectors, samples_labels, \
        samples_indices_global, samples_indices_cell_group, dirty_cell_values_cluster, n_user_labeled_cells, \
        n_model_labeled_cells, cells_with_propagated_labels_vectors, cells_with_propagated_labels_cg_idx, \
            cells_with_propagated_labels_gl_idx, y_cells_with_propagated_labels = \
            pick_samples_in_cell_cluster(cluster, 
                                         updated_cells_per_cluster, 
                                         updated_cell_cluster_n_labels, 
                                         x, 
                                         y, 
                                         dirty_cell_values, 
                                         tables_tuples_dict,
                                         nearest_neighbours_percentage,
                                         semi_propagation,
                                         labeling_method,
                                         llm_labels_per_cell_group,
                                         original_data_keys_temp, 
                                         min_n_labels_per_cell_group)
        global_n_user_labeled_cells += n_user_labeled_cells
        global_n_model_labeled_cells += n_model_labeled_cells

        cell_clustering_dict, samples_dict = update_samples_dict(cell_clustering_dict, samples_dict, cluster, \
                        samples_feature_vectors, samples_labels, samples_indices_global, \
                            samples_indices_cell_group, dirty_cell_values_cluster, \
                                updated_cells_per_cluster, updated_errors_per_cluster, updated_cell_cluster_n_labels, \
                                    cells_with_propagated_labels_vectors, cells_with_propagated_labels_cg_idx, cells_with_propagated_labels_gl_idx, y_cells_with_propagated_labels)
        
    logging.debug("Sampling done")
    
    return samples_dict, cell_clustering_dict, global_n_user_labeled_cells, global_n_model_labeled_cells



def labeling(samples_dict):
    try:
        logging.debug("Labeling")
        samples_dict.update({"final_label_to_be_propagated": []})
        for cell_cluster_idx, _ in enumerate(samples_dict["cell_cluster"]):
            if len(samples_dict["samples"][cell_cluster_idx]) != 0:
                if len(set(samples_dict["labels"][cell_cluster_idx])) == 1:
                    samples_dict["final_label_to_be_propagated"].append(
                        samples_dict["labels"][cell_cluster_idx][0]
                    )
                else:
                    logging.debug("*******Labeling - mode: %s", mode(samples_dict["labels"][cell_cluster_idx]))
                    samples_dict["final_label_to_be_propagated"].append(
                        mode(samples_dict["labels"][cell_cluster_idx])
                    )
            else:
                samples_dict["final_label_to_be_propagated"].append(None)
        logging.debug("Labeling  done")
    except Exception as e:
        logging.error("Labeling error: %s", e)
    return samples_dict
