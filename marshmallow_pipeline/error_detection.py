from collections import OrderedDict
import logging
import os
import pickle
import sys
import multiprocessing
import itertools

import pandas as pd

from marshmallow_pipeline.cell_grouping_module.extract_table_group_charset import (
    extract_charset,
)
from marshmallow_pipeline.cell_grouping_module.generate_cell_features import (
    get_cells_features,
)
from marshmallow_pipeline.cell_grouping_module.sampling_labeling import (
    cell_clustering,
    get_n_labels,
    labeling,
    sampling,
    update_n_labels,
)
from marshmallow_pipeline.classification_module.classifier import classify
from marshmallow_pipeline.classification_module.get_train_test import (
    get_train_test_sets,
)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

def cluster_column_group_init(col_groups_dir, df_n_labels, features_dict):
    global col_groups_dir_glob 
    col_groups_dir_glob = col_groups_dir

    global df_n_labels_glob
    df_n_labels_glob = df_n_labels

    global features_dict_glob
    features_dict_glob = features_dict

def test_init(df_n_labels, output_path, all_cell_clusters_records, cell_cluster_cells_dict_all):
    global df_n_labels_glob
    df_n_labels_glob = df_n_labels

    global output_path_glob
    output_path_glob = output_path

    global all_cell_clusters_records_glob
    all_cell_clusters_records_glob = all_cell_clusters_records

    global cell_cluster_cells_dict_all_glob
    cell_cluster_cells_dict_all_glob = cell_cluster_cells_dict_all


def get_cells_in_cluster(group_df, col_cluster, features_dict):
    original_data_keys_temp = []
    value_temp = []
    X_temp = []
    y_temp = []
    key_temp = []
    datacells_uids = {}
    current_local_cell_uid = 0
    all_table_cols = []

    try:
        c_df = group_df[group_df["column_cluster_label"] == col_cluster]
        for _, row in c_df.iterrows():
            all_table_cols.append((row['table_id'], row['col_id']))
        for _, row in c_df.iterrows():
            table_col_id_features = OrderedDict()
            for table_col in all_table_cols:
                table_col_id_features[table_col] = 0
            table_col_id_features[(row['table_id'], row['col_id'])] = 1
            table_col_features_list = list(table_col_id_features.values())

            for cell_idx in range(len(row["col_value"])):
                original_data_keys_temp.append(
                    (
                        row["table_id"],
                        row["col_id"],
                        cell_idx,
                        row["col_value"][cell_idx],
                    )
                )

                value_temp.append(row["col_value"][cell_idx])
                complete_feature_vector = features_dict[
                    (row['table_id'], row['col_id'], cell_idx, 'og')
                    ].tolist()
                complete_feature_vector.extend(table_col_features_list)
                X_temp.append(complete_feature_vector)                
                y_temp.append(
                    features_dict[
                        (row["table_id"], row["col_id"], cell_idx, "gt")
                    ].tolist()
                )
                key_temp.append((row["table_id"], row["col_id"], cell_idx))
                datacells_uids[
                    (
                        row["table_id"],
                        row["col_id"],
                        cell_idx,
                        row["col_value"][cell_idx],
                    )
                ] = current_local_cell_uid
                current_local_cell_uid += 1
    except Exception as e:
        logging.error("Error in cluster {}".format(str(col_cluster)))
        logging.error(e)

    cell_cluster_cells_dict = {
        "col_cluster": col_cluster,
        "original_data_keys_temp": original_data_keys_temp,
        "value_temp": value_temp,
        "X_temp": X_temp,
        "y_temp": y_temp,
        "key_temp": key_temp,
        "datacells_uids": datacells_uids,
    }
    return cell_cluster_cells_dict


def col_clu_cell_clustering(
    n_cell_clusters_per_col_cluster, table_cluster, col_cluster, group_df, features_dict, n_cores
):
    logging.debug("Processing cluster %s", str(col_cluster))
    cell_cluster_cells_dict = get_cells_in_cluster(group_df, col_cluster, features_dict)
    cell_clustering_dict = cell_clustering(
        table_cluster,
        col_cluster,
        cell_cluster_cells_dict["X_temp"],
        cell_cluster_cells_dict["y_temp"],
        n_cell_clusters_per_col_cluster,
        n_cores
    )
    logging.debug("processing cluster %s ... done", str(col_cluster))
    return cell_cluster_cells_dict, cell_clustering_dict


def cel_cluster_sampling_labeling(cell_clustering_df, cell_cluster_cells_dict):
    logging.info(
        "Sampling and labeling cluster %s",
        str(cell_clustering_df["col_cluster"].values[0]),
    )
    logging.debug(
        "Number of labels (updated): %s",
        str(cell_clustering_df["n_labels_updated"].values[0]),
    )

    try:
        if cell_clustering_df["n_labels_updated"].values[0] > 0:
            X_temp = cell_cluster_cells_dict["X_temp"]
            y_temp = cell_cluster_cells_dict["y_temp"]
            value_temp = cell_cluster_cells_dict["value_temp"]
            key_temp = cell_cluster_cells_dict["key_temp"]

            samples_dict = sampling(cell_clustering_df, X_temp, y_temp, value_temp)
            samples_dict = labeling(samples_dict)
            universal_samples = {}
            logging.debug("len samples: %s", str(len(samples_dict["cell_cluster"])))
            for cell_cluster_idx, _ in enumerate(samples_dict["cell_cluster"]):
                if len(samples_dict["samples"][cell_cluster_idx]) > 0:
                    for idx, cell_idx in enumerate(
                        samples_dict["samples_indices_global"][cell_cluster_idx]
                    ):
                        universal_samples.update(
                            {
                                key_temp[cell_idx]: samples_dict["labels"][
                                    cell_cluster_idx
                                ][idx]
                            }
                        )
            logging.debug("len to_be_added: %s", str(len(universal_samples)))
        else:
            # we need at least 2 labels per col group (in the cases that we have only one cluster 1 label is enough)
            samples_dict = None

        if samples_dict is None:
            return None
        else:
            X_labeled_by_user = []
            y_labeled_by_user = []
            for cell_cluster_idx, _ in enumerate(samples_dict["cell_cluster"]):
                if len(samples_dict["samples"][cell_cluster_idx]) > 0:
                    X_labeled_by_user.extend(samples_dict["samples"][cell_cluster_idx])
                    y_labeled_by_user.extend(samples_dict["labels"][cell_cluster_idx])
            logging.debug("len X_labeled_by_user: %s", str(len(X_labeled_by_user)))
            X_train, y_train, X_test, y_test, y_cell_ids = get_train_test_sets(
                X_temp, y_temp, samples_dict, cell_clustering_df
            )
            predicted = classify(X_train, y_train, X_test)
    except Exception as e:
        logging.error(
            "Error in cluster %s", str(cell_clustering_df["col_cluster"].values[0])
        )
        logging.error(e)

    cel_cluster_sampling_labeling_dict = {
        "y_test": y_test,
        "y_cell_ids": y_cell_ids,
        "predicted": predicted,
        "original_data_keys_temp": cell_cluster_cells_dict["original_data_keys_temp"],
        "universal_samples": universal_samples,
        "X_labeled_by_user": X_labeled_by_user,
        "y_labeled_by_user": y_labeled_by_user,
        "datacells_uids": cell_cluster_cells_dict["datacells_uids"],
    }
    logging.info(
        "Finished sampling and labeling cluster %s",
        str(cell_clustering_df["col_cluster"].values[0]),
    )
    logging.debug("Number of labels (used): %s", str(len(X_labeled_by_user)))
    return cel_cluster_sampling_labeling_dict


def error_detector(
    cell_feature_generator_enabled,
    sandbox_path,
    col_groups_dir,
    output_path,
    results_path,
    n_labels,
    number_of_col_clusters,
    cluster_sizes_dict,
    cell_clustering_alg,
    tables_dict,
    min_num_labes_per_col_cluster,
    dirty_files_name,
    clean_files_name,
    n_cores
):
    logging.info("Starting error detection")

    logging.info("Extracting table charsets")
    table_charset_dict = extract_charset(col_groups_dir)

    logging.info("Generating cell features")
    if cell_feature_generator_enabled:
        logging.info("Generating cell features enabled")
        features_dict = get_cells_features(
            sandbox_path, output_path, table_charset_dict, tables_dict, dirty_files_name, clean_files_name
        )
    else:
        logging.info("Generating cell features disabled, loading from previous results from disk")
        with open(os.path.join(output_path, "features.pickle"), "rb") as pickle_file:
            features_dict = pickle.load(pickle_file)

    logging.info("Selecting label")
    cluster_sizes_df = pd.DataFrame.from_dict(cluster_sizes_dict)
    df_n_labels = get_n_labels(
        cluster_sizes_df,
        labeling_budget=n_labels,
        min_num_labes_per_col_cluster=min_num_labes_per_col_cluster,
    )

    logging.info("Clustering column groups")
    col_group_file_names = [file_name for file_name in os.listdir(col_groups_dir) if ".pickle" in file_name]
    # n_processes = min((len(col_group_file_names), os.cpu_count()))
    # logging.debug("Number of processes: %s", str(n_processes))
    with multiprocessing.Pool(initializer=cluster_column_group_init, initargs=(col_groups_dir, df_n_labels, features_dict), processes=16) as pool:
        table_clusters = []
        cell_cluster_cells_dict_all = {}
        cell_clustering_dict_all = {}
        col_clusters = []
        logging.info("Number of column groups: %s", str(len(col_group_file_names)))
        logging.info("Starting parallel processing of column groups")
        # Prepare the arguments as tuples
        args = [(x, n_cores) for x in col_group_file_names]
        logging.debug("args: %s", str(args))
        # Use starmap to pass arguments as separate values
        pool_results = pool.starmap(cluster_column_group, args)
        logging.info("Storing cluster_column_group results")
        for result in pool_results:
            if result is not None:
                table_clusters.append(result["table_cluster"])
                cell_cluster_cells_dict_all.update(result["cell_cluster_cells_dict_all"])
                cell_clustering_dict_all.update(result["cell_clustering_dict_all"])
                col_clusters.append(result["col_clusters"])


    logging.info("Saving clustering results")
    all_cell_clusters_records = []
    for table_group in cell_clustering_dict_all:
        for col_group in cell_clustering_dict_all[table_group]:
            all_cell_clusters_records.append(cell_clustering_dict_all[table_group][col_group])

    all_cell_clusters_records = update_n_labels(all_cell_clusters_records)
    with open(
        os.path.join(output_path, "all_cell_clusters_records.pickle"), "wb"
    ) as pickle_file:
        pickle.dump(all_cell_clusters_records, pickle_file)
    with open(
        os.path.join(output_path, "cell_cluster_cells_dict_all.pickle"), "wb"
    ) as pickle_file:
        pickle.dump(cell_cluster_cells_dict_all, pickle_file)


    logging.info("Sampling and labeling clusters")
    col_clusters = []
    table_clusters = []
    for table_cluster in cell_cluster_cells_dict_all:
        for col_cluster in cell_cluster_cells_dict_all[table_cluster]:
            table_clusters.append(table_cluster)
            col_clusters.append(col_cluster)

    with multiprocessing.Pool(initializer=test_init, initargs=(df_n_labels, output_path, all_cell_clusters_records, cell_cluster_cells_dict_all), processes=min(len(cell_cluster_cells_dict_all), os.cpu_count())) as pool:
        original_data_keys = []
        unique_cells_local_index_collection = {}
        predicted_all = {}
        y_test_all = {}
        y_local_cell_ids = {}
        X_labeled_by_user_all = {}
        y_labeled_by_user_all = {}
        selected_samples = {}
        used_labels = 0
        logging.info("Starting parallel processing of clusters")
        pool_results = pool.starmap(test, zip(col_clusters, table_clusters))

        for result in pool_results:
            if result is not None:
                original_data_keys.append(result["original_data_keys"])
                unique_cells_local_index_collection.update(result["unique_cells_local_index_collection"])
                predicted_all.update(result["predicted_all"])
                y_test_all.update(result["y_test_all"])
                y_local_cell_ids.update(result["y_local_cell_ids"])
                X_labeled_by_user_all.update(result["X_labeled_by_user_all"])
                y_labeled_by_user_all.update(result["y_labeled_by_user_all"])
                selected_samples.update(result["selected_samples"])
                used_labels += result["used_labels"]

    logging.info("Saving results")
    with open(os.path.join(output_path, "original_data_keys.pkl"), "wb") as filehandler:
        pickle.dump(original_data_keys, filehandler)

    with open(os.path.join(results_path, "sampled_tuples.pkl"), "wb") as filehandler:
        pickle.dump(selected_samples, filehandler)
        logging.info("Number of Labeled Cells: %s", len(selected_samples))

    with open(os.path.join(output_path, "df_n_labels.pkl"), "wb") as filehandler:
        pickle.dump(df_n_labels, filehandler)

    return (
        y_test_all,
        y_local_cell_ids,
        predicted_all,
        y_labeled_by_user_all,
        unique_cells_local_index_collection,
        selected_samples,
    )


def cluster_column_group(file_name, n_cores):
    logging.info("Clustering column group: %s", file_name)
    table_clusters = []
    cell_cluster_cells_dict_all = {}
    cell_clustering_dict_all = {}
    col_clusters = []

    global col_groups_dir_glob 
    col_groups_dir = col_groups_dir_glob
    global df_n_labels_glob
    df_n_labels = df_n_labels_glob
    global features_dict_glob
    features_dict = features_dict_glob

    with open(os.path.join(col_groups_dir, file_name), "rb") as file:
        group_df = pickle.load(file)
        if not isinstance(group_df, pd.DataFrame):
            group_df = pd.DataFrame.from_dict(group_df, orient="index").T
        table_cluster = int(
            file_name.removeprefix("col_df_labels_cluster_").removesuffix(
                ".pickle"
            )
        )
        table_clusters.append(table_cluster)
        cell_cluster_cells_dict_all[table_cluster] = {}
        cell_clustering_dict_all[table_cluster] = {}
        file.close()
        clusters = df_n_labels[df_n_labels["table_cluster"] == table_cluster][
            "col_cluster"
        ].values
        for _, col_cluster in enumerate(clusters):
            col_clusters.append(col_cluster)
            n_cell_groups = (
                df_n_labels[
                    (df_n_labels["table_cluster"] == table_cluster)
                    & (df_n_labels["col_cluster"] == col_cluster)
                ]["n_labels"].values[0]
            )

            (
                cell_cluster_cells_dict,
                cell_clustering_dict,
            ) = col_clu_cell_clustering(
                n_cell_groups,
                table_cluster,
                col_cluster,
                group_df,
                features_dict,
                n_cores
            )
            cell_cluster_cells_dict_all[table_cluster][
                col_cluster
            ] = cell_cluster_cells_dict
            cell_clustering_dict_all[table_cluster][
                col_cluster
            ] = cell_clustering_dict

    logging.info("Clustering column group: %s ... done", file_name)

    return {"table_cluster": table_clusters, "col_clusters": col_clusters, "cell_cluster_cells_dict_all": cell_cluster_cells_dict_all, "cell_clustering_dict_all": cell_clustering_dict_all}

def test(col_cluster, table_cluster):
    logging.info("Starting test; Column cluster: %s; Table cluster %s", col_cluster, table_cluster)
    original_data_keys = []
    unique_cells_local_index_collection = {}
    predicted_all = {}
    y_test_all = {}
    y_local_cell_ids = {}
    X_labeled_by_user_all = {}
    y_labeled_by_user_all = {}
    selected_samples = {}
    used_labels = 0

    global df_n_labels_glob
    df_n_labels = df_n_labels_glob

    global output_path_glob
    output_path = output_path_glob

    global all_cell_clusters_records_glob
    all_cell_clusters_records = all_cell_clusters_records_glob

    global cell_cluster_cells_dict_all_glob
    cell_cluster_cells_dict_all = cell_cluster_cells_dict_all_glob

    cell_clustering_df = all_cell_clusters_records[
        (all_cell_clusters_records["table_cluster"] == table_cluster)
        & (all_cell_clusters_records["col_cluster"] == col_cluster)
    ]
    cell_cluster_cells_dict = cell_cluster_cells_dict_all[table_cluster][
        col_cluster
    ]
    cel_cluster_sampling_labeling_dict = cel_cluster_sampling_labeling(
        cell_clustering_df, cell_cluster_cells_dict
    )
    with open(
        os.path.join(
            output_path,
            f"cel_cluster_sampling_labeling_dict_{table_cluster}_{col_cluster}.pickle",
        ),
        "wb",
    ) as pickle_file:
        pickle.dump(cel_cluster_sampling_labeling_dict, pickle_file)

    X_labeled_by_user = cel_cluster_sampling_labeling_dict["X_labeled_by_user"]

    used_labels += len(X_labeled_by_user) if X_labeled_by_user is not None else 0
    df_n_labels.loc[
        (df_n_labels["table_cluster"] == table_cluster)
        & (df_n_labels["col_cluster"] == col_cluster),
        "sampled",
    ] = True
    if X_labeled_by_user is not None:
        selected_samples.update(
            cel_cluster_sampling_labeling_dict["universal_samples"]
        )
        original_data_keys.extend(
            cel_cluster_sampling_labeling_dict["original_data_keys_temp"]
        )

        X_labeled_by_user_all[
            (str(table_cluster), str(col_cluster))
        ] = X_labeled_by_user
        y_labeled_by_user_all[
            (str(table_cluster), str(col_cluster))
        ] = cel_cluster_sampling_labeling_dict["y_labeled_by_user"]

        predicted_all[
            (str(table_cluster), str(col_cluster))
        ] = cel_cluster_sampling_labeling_dict["predicted"]
        y_test_all[
            (str(table_cluster), str(col_cluster))
        ] = cel_cluster_sampling_labeling_dict["y_test"]
        y_local_cell_ids[
            (str(table_cluster), str(col_cluster))
        ] = cel_cluster_sampling_labeling_dict["y_cell_ids"]
        unique_cells_local_index_collection[
            (str(table_cluster), str(col_cluster))
        ] = cel_cluster_sampling_labeling_dict["datacells_uids"]

    logging.info("Done test; Column cluster: %s; Table cluster %s; Used labels %s ", col_cluster, table_cluster, str(len(X_labeled_by_user) if X_labeled_by_user is not None else 0))
    
    return {"original_data_keys": original_data_keys, "unique_cells_local_index_collection": unique_cells_local_index_collection, "predicted_all": predicted_all, "y_test_all": y_test_all, "y_local_cell_ids": y_local_cell_ids, "X_labeled_by_user_all": X_labeled_by_user_all, "y_labeled_by_user_all": y_labeled_by_user_all, "selected_samples": selected_samples, "used_labels": used_labels}
