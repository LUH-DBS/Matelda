import logging
import os
import pickle
import sys
import time

import pandas as pd
from marshmallow_pipeline.cell_grouping_module.cell_folding import cell_cluster_sampling_labeling, cluster_column_group

from marshmallow_pipeline.cell_grouping_module.extract_table_group_charset import (
    extract_charset,
)
from marshmallow_pipeline.cell_grouping_module.generate_cell_features import (
    get_cells_features,
)
from marshmallow_pipeline.cell_grouping_module.sampling_labeling import (
    get_n_labels,
    update_n_labels,
)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

def error_detector(
    cell_feature_generator_enabled,
    sandbox_path,
    col_groups_dir,
    output_path,
    results_path,
    n_labels,
    labels_per_cell_group,
    cluster_sizes_dict,
    tables_dict,
    min_num_labes_per_col_cluster,
    dirty_files_name,
    clean_files_name,
    n_cores,
    cell_clustering_res_available,
    save_mediate_res_on_disk,
    pool,
    classification_mode,
    raha_config
):
    logging.info("Starting error detection")

    logging.info("Extracting table charsets")
    table_charset_dict = extract_charset(col_groups_dir)

    logging.info("Generating cell features")
    if cell_feature_generator_enabled:
        logging.info("Generating cell features enabled")
        features_dict, tables_tuples_dict = get_cells_features(
            sandbox_path, output_path, table_charset_dict, tables_dict, dirty_files_name, clean_files_name, save_mediate_res_on_disk, pool, raha_config
        )
    else:
        logging.info("Generating cell features disabled, loading from previous results from disk")
        with open(os.path.join(output_path, "features.pickle"), "rb") as pickle_file:
            features_dict = pickle.load(pickle_file)
        with open(os.path.join(output_path, "tables_tuples.pickle"), "rb") as pickle_file:
            tables_tuples_dict = pickle.load(pickle_file)

    logging.info("Selecting label")
    cluster_sizes_df = pd.DataFrame.from_dict(cluster_sizes_dict)
    df_n_labels = get_n_labels(
        cluster_sizes_df,
        labeling_budget=n_labels,
        min_num_labes_per_col_cluster=min_num_labes_per_col_cluster,
    )

    if not cell_clustering_res_available:
        logging.info("Cell Clustering")
        start_time = time.time()
        col_group_file_names = [file_name for file_name in os.listdir(col_groups_dir) if ".pickle" in file_name]
        n_processes = min((len(col_group_file_names), os.cpu_count()))
        # logging.debug("Number of processes: %s", str(n_processes))

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
        results = []
        for x in col_group_file_names:
            results.append(cluster_column_group(col_groups_dir, df_n_labels, features_dict, labels_per_cell_group, x, n_cores))
        logging.info("Storing cluster_column_group results")
        for result in results:
            if result is not None:
                table_clusters.append(result["table_cluster"])
                cell_cluster_cells_dict_all.update(result["cell_cluster_cells_dict_all"])
                cell_clustering_dict_all.update(result["cell_clustering_dict_all"])
                col_clusters.append(result["col_clusters"])


        all_cell_clusters_records = []
        for table_group in cell_clustering_dict_all:
            for col_group in cell_clustering_dict_all[table_group]:
                all_cell_clusters_records.append(cell_clustering_dict_all[table_group][col_group])

        all_cell_clusters_records = update_n_labels(all_cell_clusters_records)
        cell_clustering_dir = os.path.join(output_path, "cell_clustering")
        end_time = time.time()
        logging.info("Cell Clustering took %s seconds", str(end_time - start_time))
        if save_mediate_res_on_disk:
            logging.info("Saving cell clustering results")
            if not os.path.exists(cell_clustering_dir):
                os.makedirs(cell_clustering_dir)
            with open(
                os.path.join(cell_clustering_dir, "all_cell_clusters_records.pickle"), "wb"
            ) as pickle_file:
                pickle.dump(all_cell_clusters_records, pickle_file)
            with open(
                os.path.join(cell_clustering_dir, "cell_cluster_cells_dict_all.pickle"), "wb"
            ) as pickle_file:
                pickle.dump(cell_cluster_cells_dict_all, pickle_file)
    else:
        logging.info("Loading cell clustering results from disk")
        with open(
            os.path.join(output_path, "cell_clustering", "all_cell_clusters_records.pickle"), "rb"
        ) as pickle_file:
            all_cell_clusters_records = pickle.load(pickle_file)
        with open(
            os.path.join(output_path, "cell_clustering", "cell_cluster_cells_dict_all.pickle"), "rb"
        ) as pickle_file:
            cell_cluster_cells_dict_all = pickle.load(pickle_file)

    logging.info("Sampling and labeling clusters")
    start_time = time.time()    
    original_data_keys = []
    unique_cells_local_index_collection = {}
    predicted_all = {}
    y_test_all = {}
    y_local_cell_ids = {}
    X_labeled_by_user_all = {}
    y_labeled_by_user_all = {}
    selected_samples = {}
    used_labels = 0
    logging.info("Starting processing of cell clusters")
    results = []
    for table_cluster in cell_cluster_cells_dict_all:
        for col_cluster in cell_cluster_cells_dict_all[table_cluster]:
            result = test(df_n_labels, output_path, all_cell_clusters_records, cell_cluster_cells_dict_all, n_cores, save_mediate_res_on_disk, classification_mode, tables_tuples_dict, labels_per_cell_group, col_cluster, table_cluster)
            results.append(result)
    n_user_labeled_cells = 0
    for result in results:
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
            n_user_labeled_cells += result["n_user_labeled_cells"]
            logging.info("Number of used Labeled Cells: %s", str(result["used_labels"]))
            logging.info("Len selected_samples: %s", str(len(result["selected_samples"])))
            logging.info("Number of Labeled Cells (user): %s", str(result["n_user_labeled_cells"]))

    end_time = time.time()
    logging.info("Sampling and labeling clusters took %s seconds", str(end_time - start_time))
    logging.info("Saving results")
    if save_mediate_res_on_disk:
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
        n_user_labeled_cells,
    )


def test(df_n_labels, output_path, all_cell_clusters_records, cell_cluster_cells_dict_all, n_cores, save_mediate_res_on_disk, classification_mode, tables_tuples_dict, labels_per_cell_group, col_cluster, table_cluster):
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

    cell_clustering_df = all_cell_clusters_records[
        (all_cell_clusters_records["table_cluster"] == table_cluster)
        & (all_cell_clusters_records["col_cluster"] == col_cluster)
    ]
    cell_cluster_cells_dict = cell_cluster_cells_dict_all[table_cluster][
        col_cluster
    ]
    cell_cluster_sampling_labeling_dict, cell_clustering_df, samples_dict, n_user_labeled_cells = cell_cluster_sampling_labeling(
        cell_clustering_df, cell_cluster_cells_dict, n_cores, classification_mode, tables_tuples_dict, labels_per_cell_group, output_path
    )

    if save_mediate_res_on_disk:
        cell_clustering_dir = os.path.join(output_path, "cell_clustering")
        if not os.path.exists(cell_clustering_dir):
            os.makedirs(cell_clustering_dir)
        with open(
            os.path.join(
                cell_clustering_dir,
                f"cell_cluster_sampling_labeling_dict_{table_cluster}_{col_cluster}.pickle",
            ),
            "wb",
        ) as pickle_file:
            pickle.dump(cell_cluster_sampling_labeling_dict, pickle_file)

        with open(
            os.path.join(
                cell_clustering_dir,
                f"samples_dict_{table_cluster}_{col_cluster}.pickle",
            ),
            "wb",
        ) as pickle_file:
            pickle.dump(samples_dict, pickle_file)
        
        with open(
            os.path.join(
                cell_clustering_dir,
                f"cell_clustering_df_{table_cluster}_{col_cluster}.pickle",
            ),
            "wb",
        ) as pickle_file:
            pickle.dump(cell_clustering_df, pickle_file)

    X_labeled_by_user = cell_cluster_sampling_labeling_dict["X_labeled_by_user"]

    used_labels += len(X_labeled_by_user) if X_labeled_by_user is not None else 0
    df_n_labels.loc[
        (df_n_labels["table_cluster"] == table_cluster)
        & (df_n_labels["col_cluster"] == col_cluster),
        "sampled",
    ] = True
    if X_labeled_by_user is not None:
        selected_samples.update(
            cell_cluster_sampling_labeling_dict["universal_samples"]
        )
        original_data_keys.extend(
            cell_cluster_sampling_labeling_dict["original_data_keys_temp"]
        )

        X_labeled_by_user_all[
            (str(table_cluster), str(col_cluster))
        ] = X_labeled_by_user
        y_labeled_by_user_all[
            (str(table_cluster), str(col_cluster))
        ] = cell_cluster_sampling_labeling_dict["y_labeled_by_user"]

        predicted_all[
            (str(table_cluster), str(col_cluster))
        ] = cell_cluster_sampling_labeling_dict["predicted"]
        y_test_all[
            (str(table_cluster), str(col_cluster))
        ] = cell_cluster_sampling_labeling_dict["y_test"]
        y_local_cell_ids[
            (str(table_cluster), str(col_cluster))
        ] = cell_cluster_sampling_labeling_dict["y_cell_ids"]
        unique_cells_local_index_collection[
            (str(table_cluster), str(col_cluster))
        ] = cell_cluster_sampling_labeling_dict["datacells_uids"]

    init_labels_tg_cg = df_n_labels[(df_n_labels["table_cluster"] == table_cluster) & (df_n_labels["col_cluster"] == col_cluster)]["n_labels"].values[0]

    logging.info("Done test; Column cluster: %s; Table cluster %s; Used labels %s , Init labels: %s", col_cluster, table_cluster, str(len(X_labeled_by_user) if X_labeled_by_user is not None else 0), init_labels_tg_cg)
    
    return {"original_data_keys": original_data_keys, "unique_cells_local_index_collection": unique_cells_local_index_collection, "predicted_all": predicted_all, "y_test_all": y_test_all, "y_local_cell_ids": y_local_cell_ids, "X_labeled_by_user_all": X_labeled_by_user_all, "y_labeled_by_user_all": y_labeled_by_user_all, "selected_samples": selected_samples, "used_labels": used_labels, "n_user_labeled_cells": n_user_labeled_cells}
