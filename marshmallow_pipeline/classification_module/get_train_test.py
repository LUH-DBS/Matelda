import copy
import logging
import time
from marshmallow_pipeline.classification_module.classifier import classify

import pandas as pd


def get_train_test_sets(X_temp, y_temp, samples_dict, cell_clustering_df):
    logging.debug("Train-Test set preparation")
    cells_per_cluster = cell_clustering_df["cells_per_cluster"].values[0]
    samples_df = pd.DataFrame(samples_dict)
    X_train, y_train, X_test, y_test, y_cell_ids = [], [], [], [], []
    clusters = samples_df["cell_cluster"].unique().tolist()
    clusters.sort()
    for key in clusters:
        try:
            if key == -1:
                continue
            cell_cluster_samples = samples_df[samples_df["cell_cluster"] == key][
                "samples_indices_global"
            ].values[0]
            cell_cluster_final_label = samples_df[samples_df["cell_cluster"] == key][
                "final_label_to_be_propagated"
            ].values[0]
            if len(cell_cluster_samples) == 0:
                for cell in cells_per_cluster[key]:
                    X_test.append(X_temp[cell])
                    y_test.append(y_temp[cell])
                    y_cell_ids.append(cell)
            else:
                for cell in cells_per_cluster[key]:
                    X_train.append(X_temp[cell])
                    if cell in cell_cluster_samples:
                        y_train.append(y_temp[cell])
                    else:
                        y_train.append(cell_cluster_final_label)
                    X_test.append(X_temp[cell])
                    y_test.append(y_temp[cell])
                    y_cell_ids.append(cell)
        except Exception as e:
            logging.error("Error in get_train_test_sets: %s", e)

    logging.debug("Length of X_train: %s", len(X_train))
    return X_train, y_train, X_test, y_test, y_cell_ids

def get_train_test_sets_per_col(X_temp, y_temp, samples_dict, cell_clustering_df, uids):
    logging.debug("Train-Test set preparation")
    cells_per_cluster = cell_clustering_df["cells_per_cluster"].values[0]
    samples_df = pd.DataFrame(samples_dict)
    uids_per_col = {}
    cols_of_uids = {}
    for uid in uids:
        if (uid[0], uid[1]) not in uids_per_col:
            uids_per_col[(uid[0], uid[1])] = {uids[uid]: uid}
        else:
            uids_per_col[(uid[0], uid[1])][uids[uid]] = uid
        cols_of_uids[uids[uid]] = (uid[0], uid[1])
    X_train_cols = {}
    y_train_cols = {}
    X_test_cols = {}
    y_test_cols = {}
    y_cell_ids_cols = {}
    predicted_cols = {}
    X_train, y_train, X_test, y_test, y_cell_ids, predicted = [], [], [], [], [], []    
    clusters = samples_df["cell_cluster"].unique().tolist()
    logging.debug("Clusters: %s", clusters)
    s_time = time.time()
    for key in clusters:
        try:
            if key == -1:
                continue
            cell_cluster_samples = samples_df[samples_df["cell_cluster"] == key][
                "samples_indices_global"
            ].values[0]
            cell_cluster_final_label = samples_df[samples_df["cell_cluster"] == key][
                "final_label_to_be_propagated"
            ].values[0]
            if len(cell_cluster_samples) == 0:
                for cell in cells_per_cluster[key]:
                    cell_col = cols_of_uids[cell]
                    if cell_col not in X_test_cols:
                        X_test_cols[cell_col] = [X_temp[cell]]
                        y_test_cols[cell_col] = [y_temp[cell]]
                        y_cell_ids_cols[cell_col] = [cell]
                    else:
                        X_test_cols[cell_col].append(X_temp[cell])
                        y_test_cols[cell_col].append(y_temp[cell])
                        y_cell_ids_cols[cell_col].append(cell)
            else:
                for cell in cells_per_cluster[key]:
                    cell_col = cols_of_uids[cell]
                    if cell_col not in X_train_cols:
                        X_train_cols[cell_col] = [X_temp[cell]]
                    else:
                        X_train_cols[cell_col].append(X_temp[cell])
                    if cell in cell_cluster_samples:
                        if cell_col not in y_train_cols:
                            y_train_cols[cell_col] = [y_temp[cell]]
                        else:
                            y_train_cols[cell_col].append(y_temp[cell])
                    else:
                        if cell_col not in y_train_cols:
                            y_train_cols[cell_col] = [cell_cluster_final_label]
                        else:
                            y_train_cols[cell_col].append(cell_cluster_final_label)
                    if cell_col not in X_test_cols:
                        X_test_cols[cell_col] = [X_temp[cell]]
                        y_test_cols[cell_col] = [y_temp[cell]]
                        y_cell_ids_cols[cell_col] = [cell]
                    else:
                        X_test_cols[cell_col].append(X_temp[cell])
                        y_test_cols[cell_col].append(y_temp[cell])
                        y_cell_ids_cols[cell_col].append(cell)
        except Exception as e:
            logging.error("Error in get_train_test_sets: %s", e)
    logging.debug("*******Time for train-test set preparation: %s", time.time() - s_time)
    s_time = time.time()
    logging.debug("Start classification Per Column")
    for col in X_train_cols:
        gbc, predicted_cols[col] = classify(X_train_cols[col], y_train_cols[col], X_test_cols[col])
    logging.debug("End classification Per Column")
    logging.debug("*******Time for classification Per Column: %s", time.time() - s_time)
    for col in predicted_cols:
        for i in range(len(predicted_cols[col])):
            X_test.append(X_test_cols[col][i])
            y_test.append(y_test_cols[col][i])
            y_cell_ids.append(y_cell_ids_cols[col][i])
            y_train.append(predicted_cols[col][i])
            X_train.append(X_train_cols[col][i])
            predicted.append(predicted_cols[col][i])
    logging.debug("Length of X_train: %s", len(X_train))
    return X_train, y_train, X_test, y_test, y_cell_ids, predicted

def get_train_test_sets_per_col_pseudo(X_temp, y_temp, samples_dict, cell_clustering_df, uids):
    logging.debug("Train-Test set preparation")
    cells_per_cluster = cell_clustering_df["cells_per_cluster"].values[0]
    samples_df = pd.DataFrame(samples_dict)
    uids_per_col = {}
    cols_of_uids = {}
    for uid in uids:
        if (uid[0], uid[1]) not in uids_per_col:
            uids_per_col[(uid[0], uid[1])] = {uids[uid]: uid}
        else:
            uids_per_col[(uid[0], uid[1])][uids[uid]] = uid
        cols_of_uids[uids[uid]] = (uid[0], uid[1])
    X_train_pseudo_cols, y_train_pseudo_cols, X_test_pseudo_cols, y_test_pseudo_cols, predicted_pseudo_cols = {}, {}, {}, {}, {}
    X_test_cols, predicted_test_cols, predicted_cols = {}, {}, {}
    y_cell_ids_pseudo_cols, y_cell_ids_cols, sample_cells_id_cols = {}, {}, {}

    clusters = samples_df["cell_cluster"].unique().tolist()
    logging.debug("Clusters: %s", clusters)
    s_time = time.time()
    for key in clusters:
        try:
            if key == -1:
                continue
            cell_cluster_samples = samples_df[samples_df["cell_cluster"] == key][
                "samples_indices_global"
            ].values[0]
            if len(cell_cluster_samples) == 0:
                for cell in cells_per_cluster[key]:
                    cell_col = cols_of_uids[cell]
                    if cell_col not in X_test_pseudo_cols:
                        X_test_pseudo_cols[cell_col] = [X_temp[cell]]
                        y_test_pseudo_cols[cell_col] = [y_temp[cell]]
                        y_cell_ids_pseudo_cols[cell_col] = [cell]
                    else:
                        X_test_pseudo_cols[cell_col].append(X_temp[cell])
                        y_test_pseudo_cols[cell_col].append(y_temp[cell])
                        y_cell_ids_pseudo_cols[cell_col].append(cell)
            else:
                for cell in cells_per_cluster[key]:
                    cell_col = cols_of_uids[cell]
                    if cell in cell_cluster_samples:
                        if cell_col not in sample_cells_id_cols:
                            sample_cells_id_cols[cell_col] = [cell]
                        else:
                            sample_cells_id_cols[cell_col].append(cell)
                        if cell_col not in X_train_pseudo_cols:
                            X_train_pseudo_cols[cell_col] = [X_temp[cell]]
                        else:
                            X_train_pseudo_cols[cell_col].append(X_temp[cell])
                        if cell_col not in y_train_pseudo_cols:
                            y_train_pseudo_cols[cell_col] = [y_temp[cell]]
                        else:
                            y_train_pseudo_cols[cell_col].append(y_temp[cell])
                    else:
                        if cell_col not in X_test_pseudo_cols:
                            X_test_pseudo_cols[cell_col] = [X_temp[cell]]
                            y_test_pseudo_cols[cell_col] = [y_temp[cell]]
                            y_cell_ids_pseudo_cols[cell_col] = [cell]
                        else:
                            X_test_pseudo_cols[cell_col].append(X_temp[cell])
                            y_test_pseudo_cols[cell_col].append(y_temp[cell])
                            y_cell_ids_pseudo_cols[cell_col].append(cell)
        except Exception as e:
            logging.error("Error in get_train_test_sets: %s", e)
    logging.debug("*******Time for train-test set preparation: %s", time.time() - s_time)
    s_time = time.time()
    logging.debug("Start classification for pseudo labeling")
    X_train_pseudo_all = []
    y_train_pseudo_all = []
    X_test_pseudo_all = []
    for col in X_train_pseudo_cols:
        X_train_pseudo_all.extend(X_train_pseudo_cols[col])
        y_train_pseudo_all.extend(y_train_pseudo_cols[col])
    for col in X_test_pseudo_cols:
        X_test_pseudo_all.extend(X_test_pseudo_cols[col])
    if sum(y_train_pseudo_all) == 0:
        predicted_pseudo_all = [0] * len(X_test_pseudo_all)
    elif sum(y_train_pseudo_all) == len(y_train_pseudo_all):
        predicted_pseudo_all = [1] * len(X_test_pseudo_all)
    else:
        gbc, predicted_pseudo_all = classify(X_train_pseudo_all, y_train_pseudo_all, X_test_pseudo_all)
    
    predicted_pseudo_all_copy = [int(x) for x in predicted_pseudo_all]
    predicted_pseudo_cols = {}

    for col in X_test_pseudo_cols:
        predicted_pseudo_cols[col] = predicted_pseudo_all_copy[:len(X_test_pseudo_cols[col])]
        predicted_pseudo_all_copy = predicted_pseudo_all_copy[len(X_test_pseudo_cols[col]):]
    X_train = []
    y_train = []
    X_test = []
    y_test = [] 
    y_cell_ids = []
    predicted = []
    X_test_cols = X_test_pseudo_cols
    predicted_test_cols = predicted_pseudo_cols
    y_cell_ids_cols = y_cell_ids_pseudo_cols

    for col in X_train_pseudo_cols:
        if col not in X_test_cols:
            X_test_cols[col] = [X_train_pseudo_cols[col]]
            predicted_test_cols[col] = [y_train_pseudo_cols[col]]
            y_test_pseudo_cols[col] = [y_train_pseudo_cols[col]]
            y_cell_ids_cols[col] = [sample_cells_id_cols[col]]
        else:
            X_test_cols[col].extend(X_train_pseudo_cols[col])
            predicted_test_cols[col].extend(y_train_pseudo_cols[col])
            y_test_pseudo_cols[col].extend(y_train_pseudo_cols[col])
            y_cell_ids_cols[col].extend(sample_cells_id_cols[col])

    for col in X_test_cols:
        gbc, predicted_cols[col] = classify(X_test_cols[col], predicted_test_cols[col], X_test_cols[col])

    for col in predicted_cols:
        for i in range(len(predicted_cols[col])):
            X_test.append(X_test_cols[col][i])
            y_test.append(y_test_pseudo_cols[col][i])
            y_cell_ids.append(y_cell_ids_cols[col][i])
            y_train.append(predicted_cols[col][i])
            X_train.append(X_test_cols[col][i])
            predicted.append(predicted_cols[col][i])
    logging.debug("Length of X_train: %s", len(X_train))
    return X_train, y_train, X_test, y_test, y_cell_ids, predicted