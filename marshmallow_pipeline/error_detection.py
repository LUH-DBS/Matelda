
import logging
import os
import pickle
import pandas as pd
import sys

from cell_grouping_module.extract_table_group_charset import extract_charset
from cell_grouping_module.generate_cell_features import get_cells_features
from cell_grouping_module.sampling_labeling import get_n_labels, cell_clustering, labeling, update_n_labels, sampling
from classification_module.classifier import classify
from classification_module.get_train_test import get_train_test_sets


logger = logging.getLogger()


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

def get_cells_in_cluster(group_df, col_cluster, features_dict):    
    original_data_keys_temp = []
    value_temp = []
    X_temp = []
    y_temp = []
    key_temp = []
    datacells_uids = {}
    current_local_cell_uid = 0
    try:
        c_df = group_df[group_df['column_cluster_label'] == col_cluster]
        for index, row in c_df.iterrows():
            for cell_idx in range(len(row['col_value'])):
                original_data_keys_temp.append(
                    (row['table_id'], row['col_id'], cell_idx, row['col_value'][cell_idx]))

                value_temp.append(row['col_value'][cell_idx])
                X_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'og')].tolist())
                y_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'gt')].tolist())
                value_temp.append(row['col_value'][cell_idx])
                key_temp.append((row['table_id'], row['col_id'], cell_idx))
                datacells_uids[(row['table_id'], row['col_id'], cell_idx, row['col_value'][cell_idx])] = current_local_cell_uid
                current_local_cell_uid += 1
    except Exception as e:
        logger.info("Error in cluster {}".format(str(col_cluster)))
        logger.error(e)

    cell_cluster_cells_dict = {
        "col_cluster": col_cluster,
        "original_data_keys_temp": original_data_keys_temp, 
        "value_temp": value_temp,
        "X_temp": X_temp,
        "y_temp": y_temp,
        "key_temp": key_temp,
        "datacells_uids": datacells_uids
    }
    return cell_cluster_cells_dict


def col_clu_cell_clustering(n_cell_clusters_per_col_cluster, table_cluster, col_cluster,\
                         group_df, features_dict):
    logger.info("Processing cluster {}".format(str(col_cluster)))
    cell_cluster_cells_dict = get_cells_in_cluster(group_df, col_cluster, features_dict)
    cell_clustering_dict = cell_clustering(table_cluster, col_cluster, cell_cluster_cells_dict["X_temp"], 
                                                                        cell_cluster_cells_dict["y_temp"], n_cell_clusters_per_col_cluster)
    return cell_cluster_cells_dict, cell_clustering_dict

def cel_cluster_sampling_labeling(cell_clustering_df, cell_cluster_cells_dict):
    logger.info("Sampling and labeling cluster {}".format(str(cell_clustering_df["col_cluster"].values[0])))
    logger.info("Number of labels (updated): {}".format(str(cell_clustering_df["n_labels_updated"].values[0])))
    if (str(cell_clustering_df["col_cluster"].values[0]) == "2"):
        print("DEBUG")
    try:
        if cell_clustering_df["n_labels_updated"].values[0] > 0:
            X_temp = cell_cluster_cells_dict["X_temp"]
            y_temp = cell_cluster_cells_dict["y_temp"]
            value_temp = cell_cluster_cells_dict["value_temp"]
            key_temp = cell_cluster_cells_dict["key_temp"]

            samples_dict = sampling(cell_clustering_df, X_temp, y_temp, value_temp)
            samples_dict = labeling(samples_dict)
            universal_samples = dict()
            logger.info("len samples: {}".format(str(len(samples_dict["cell_cluster"]))))
            for cell_cluster_idx, cell_cluster in enumerate(samples_dict["cell_cluster"]):
                for idx, cell_idx in enumerate(samples_dict["samples_indices_global"][cell_cluster_idx]):
                     universal_samples.update({key_temp[cell_idx]: samples_dict["labels"][cell_cluster_idx][idx]})
            logger.info("len to_be_added: {}".format(str(len(universal_samples))))
        else:
            # we need at least 2 labels per col group (in the cases that we have only one cluster 1 label is enough)
            samples_dict = None
        
        if samples_dict is None:
            return None
        else:
            X_labeled_by_user = []
            y_labeled_by_user = []
            for cell_cluster_idx, cell_cluster in enumerate(samples_dict["cell_cluster"]):
                X_labeled_by_user.extend(samples_dict["samples"][cell_cluster_idx])
                y_labeled_by_user.extend(samples_dict["labels"][cell_cluster_idx])
            logger.info("len X_labeled_by_user: {}".format(str(len(X_labeled_by_user))))
            X_train, y_train, X_test, y_test, y_cell_ids = get_train_test_sets(X_temp, y_temp, samples_dict, cell_clustering_df)
            predicted = classify(X_train, y_train, X_test)
    except Exception as e:
        logger.info("Error in cluster {}".format(str(cell_clustering_df["col_cluster"].values[0])))
        logger.error(e)

    cel_cluster_sampling_labeling_dict = {
        "y_test": y_test,
        "y_cell_ids": y_cell_ids,
        "predicted": predicted,
        "original_data_keys_temp": cell_cluster_cells_dict["original_data_keys_temp"],
        "universal_samples": universal_samples,
        "X_labeled_by_user": X_labeled_by_user,
        "y_labeled_by_user": y_labeled_by_user,
        "datacells_uids": cell_cluster_cells_dict["datacells_uids"]
    }
    logger.info("Finished sampling and labeling cluster {}".format(str(cell_clustering_df["col_cluster"].values[0])))
    logger.info("Number of labels (used): {}".format(str(len(X_labeled_by_user))))
    if len(X_labeled_by_user) != cell_clustering_df["n_labels_updated"].values[0]:
        logger.info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&I'm here$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    return cel_cluster_sampling_labeling_dict


def error_detector(cell_feature_generator_enabled, sandbox_path, col_groups_dir, 
                   output_path, results_path, n_labels, number_of_col_clusters, 
                   cluster_sizes_dict, cell_clustering_alg, tables_dict):

    logger.info("Starting error detection")
    original_data_keys = []
    unique_cells_local_index_collection = dict()
    predicted_all = dict()
    y_test_all = dict()
    y_local_cell_ids = dict()
    X_labeled_by_user_all = dict()
    y_labeled_by_user_all = dict()
    selected_samples = dict()
    used_labels = 0

    table_charset_dict = extract_charset(col_groups_dir)

    if cell_feature_generator_enabled:
        features_dict = get_cells_features(sandbox_path, output_path, table_charset_dict, tables_dict)
        logger.info("Generating cell features started.")
    else:
        with open(os.path.join(output_path, "features.pickle"), 'rb') as pickle_file:
            features_dict = pickle.load(pickle_file)

    cluster_sizes_df = pd.DataFrame.from_dict(cluster_sizes_dict)
    df_n_labels = get_n_labels(cluster_sizes_df, labeling_budget=n_labels)
    table_clusters = []
    col_clusters = []
    cell_cluster_cells_dict_all = {}
    cell_clustering_dict_all = {}

    for file_name in os.listdir(col_groups_dir):
        if ".pickle" in file_name:
            file = open(os.path.join(col_groups_dir, file_name), 'rb')
            group_df = pickle.load(file)
            if not isinstance(group_df, pd.DataFrame):
                group_df = pd.DataFrame.from_dict(group_df, orient='index').T
            table_cluster = int(file_name.removeprefix("col_df_labels_cluster_").removesuffix(".pickle"))
            table_clusters.append(table_cluster)
            cell_cluster_cells_dict_all[table_cluster] = {}
            cell_clustering_dict_all[table_cluster] = {}
            file.close()
            clusters = df_n_labels[df_n_labels['table_cluster'] == table_cluster]['col_cluster'].values
            for c_idx, col_cluster in enumerate(clusters):
                col_clusters.append(col_cluster)
                n_cell_groups = df_n_labels[(df_n_labels['table_cluster'] == table_cluster) & (df_n_labels['col_cluster'] == col_cluster)]['n_labels'].values[0] + 1
                
                cell_cluster_cells_dict, cell_clustering_dict = \
                    col_clu_cell_clustering(n_cell_groups, table_cluster, col_cluster, group_df, features_dict)
                cell_cluster_cells_dict_all[table_cluster][col_cluster] = cell_cluster_cells_dict
                cell_clustering_dict_all[table_cluster][col_cluster] = cell_clustering_dict

    all_cell_clusters_records = []
    for table_group in cell_clustering_dict_all:
        for col_group in cell_clustering_dict_all[table_group]:
            all_cell_clusters_records.append(cell_clustering_dict_all[table_group][col_group])
    all_cell_clusters_records = update_n_labels(all_cell_clusters_records)

    for table_cluster in cell_cluster_cells_dict_all:
        for col_cluster in cell_cluster_cells_dict_all[table_cluster]:
            cell_clustering_df = all_cell_clusters_records[(all_cell_clusters_records['table_cluster'] == table_cluster) & (all_cell_clusters_records['col_cluster'] == col_cluster)]
            cell_cluster_cells_dict = cell_cluster_cells_dict_all[table_cluster][col_cluster]
            cel_cluster_sampling_labeling_dict = cel_cluster_sampling_labeling(cell_clustering_df, cell_cluster_cells_dict)

            X_labeled_by_user = cel_cluster_sampling_labeling_dict["X_labeled_by_user"]
            
            used_labels += len(X_labeled_by_user) if X_labeled_by_user is not None else 0
            df_n_labels.loc[(df_n_labels['table_cluster'] == table_cluster) & (df_n_labels['col_cluster'] == col_cluster), 'sampled'] = True
            if X_labeled_by_user is not None:
                selected_samples.update(cel_cluster_sampling_labeling_dict["universal_samples"])
                original_data_keys.extend( cel_cluster_sampling_labeling_dict["original_data_keys_temp"])

                X_labeled_by_user_all[(str(table_cluster), str(col_cluster))] = X_labeled_by_user
                y_labeled_by_user_all[(str(table_cluster), str(col_cluster))] =  cel_cluster_sampling_labeling_dict["y_labeled_by_user"]
                
                predicted_all[(str(table_cluster), str(col_cluster))] = cel_cluster_sampling_labeling_dict["predicted"]
                y_test_all[(str(table_cluster), str(col_cluster))] = cel_cluster_sampling_labeling_dict["y_test"]
                y_local_cell_ids[(str(table_cluster), str(col_cluster))] = cel_cluster_sampling_labeling_dict["y_cell_ids"]
                unique_cells_local_index_collection[(str(table_cluster), str(col_cluster))] = cel_cluster_sampling_labeling_dict["datacells_uids"]

            logging.info("%%%%%%%%%%%%%%%%%%done - Processing col cluster {} table cluster {}, used labels {}".format(str(col_cluster), str(table_cluster), str(len(X_labeled_by_user) if X_labeled_by_user is not None else 0)))

    with open(os.path.join(output_path, "original_data_keys.pkl"), "wb") as filehandler:
        pickle.dump(original_data_keys, filehandler)

    with open(os.path.join(results_path, "sampled_tuples.pkl"), "wb") as filehandler:
        pickle.dump(selected_samples, filehandler)
        logger.info("Number of Labeled Cells: {}".format(len(selected_samples)))
        
    with open(os.path.join(output_path, "df_n_labels.pkl"), "wb") as filehandler:
        pickle.dump(df_n_labels, filehandler)

    return y_test_all, y_local_cell_ids, predicted_all, y_labeled_by_user_all,\
                unique_cells_local_index_collection, selected_samples
