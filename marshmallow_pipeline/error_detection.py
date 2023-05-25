
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


def process_col_cluster(n_cell_clusters_per_col_cluster, table_cluster, col_cluster,\
                         group_df, features_dict):
    X_train = []
    y_train = []
    X_temp = []
    y_temp = []
    key_temp = []
    value_temp = []
    original_data_keys_temp = []
    X_labeled_by_user = []
    y_labeled_by_user = []
    current_local_cell_uid = 0
    datacells_uids = dict()
    logger.info("Processing cluster {}".format(str(col_cluster)))

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
        
        cell_clustering_df = cell_clustering(table_cluster, col_cluster, X_temp, y_temp, n_cell_clusters_per_col_cluster)
        cell_clustering_df = update_n_labels(cell_clustering_df)
        if cell_clustering_df["n_labels_updated"].values[0] > 0:
            samples_dict = sampling(cell_clustering_df, X_temp, y_temp, value_temp)
            samples_dict = labeling(samples_dict)
            for cell_cluster_idx, cell_cluster in enumerate(samples_dict["cell_cluster"]):
                universal_samples = {key_temp[cell_idx]: samples_dict["labels"][cell_cluster_idx][idx] for idx, cell_idx in enumerate(samples_dict["samples_indices"][cell_cluster_idx])}
                
        else:
            # we need at least 2 labels per col group (in the cases that we have only one cluster 1 label is enough)
            samples_dict = None
        
        if samples_dict is None:
            return None, None, None, None, None, None, None, None
        else:
            for cell_cluster_idx, cell_cluster in enumerate(samples_dict["cell_cluster"]):
                X_labeled_by_user.extend(samples_dict["samples"][cell_cluster_idx])
                y_labeled_by_user.extend(samples_dict["labels"][cell_cluster_idx])
            X_train, y_train, X_test, y_test, y_cell_ids = get_train_test_sets(X_temp, y_temp, samples_dict, cell_clustering_df)
            predicted = classify(X_train, y_train, X_test)
    except Exception as e:
        logger.error(e)
    
    return y_test, y_cell_ids, predicted, original_data_keys_temp, universal_samples, X_labeled_by_user, y_labeled_by_user, datacells_uids


def error_detector(cell_feature_generator_enabled, extract_cell_clusters_enabled, sandbox_path, col_groups_dir, 
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

    for file_name in os.listdir(col_groups_dir):
        if ".pickle" in file_name:
            file = open(os.path.join(col_groups_dir, file_name), 'rb')
            group_df = pickle.load(file)
            if not isinstance(group_df, pd.DataFrame):
                group_df = pd.DataFrame.from_dict(group_df, orient='index').T
            table_cluster = int(file_name.removeprefix("col_df_labels_cluster_").removesuffix(".pickle"))
            file.close()
            clusters = df_n_labels[df_n_labels['table_cluster'] == table_cluster]['col_cluster'].values
            for c_idx, cluster in enumerate(clusters):
                n_cell_groups = df_n_labels[(df_n_labels['table_cluster'] == table_cluster) & (df_n_labels['col_cluster'] == cluster)]['n_labels'].values[0] + 1
                
                y_test, y_cell_ids, predicted, original_data_keys_temp, universal_samples, \
                X_labeled_by_user, y_labeled_by_user, datacells_local_ids = \
                    process_col_cluster(n_cell_groups, table_cluster, cluster, group_df, features_dict)
                used_labels += len(X_labeled_by_user) if X_labeled_by_user is not None else 0
                df_n_labels.loc[(df_n_labels['table_cluster'] == table_cluster) & (df_n_labels['col_cluster'] == cluster), 'sampled'] = True
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
        
    with open(os.path.join(output_path, "df_n_labels.pkl"), "wb") as filehandler:
        pickle.dump(df_n_labels, filehandler)

    return y_test_all, y_local_cell_ids, predicted_all, y_labeled_by_user_all,\
                unique_cells_local_index_collection, selected_samples
    