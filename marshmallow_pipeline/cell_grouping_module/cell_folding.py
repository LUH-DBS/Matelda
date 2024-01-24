import logging
import math
import os
import pickle

import pandas as pd
from marshmallow_pipeline.cell_grouping_module.generate_cell_features import get_cells_in_cluster
from marshmallow_pipeline.cell_grouping_module.sampling_labeling import cell_clustering, labeling, sampling
from marshmallow_pipeline.classification_module.classifier import classify
from marshmallow_pipeline.classification_module.get_train_test import get_train_test_sets, get_train_test_sets_per_col


def col_clu_cell_clustering(
    n_cell_clusters_per_col_cluster, table_cluster, col_cluster, group_df, features_dict, n_cores, labels_per_cell_group
):
    logging.debug("Processing cluster %s", str(col_cluster))
    cell_cluster_cells_dict = get_cells_in_cluster(group_df, col_cluster, features_dict)
    cell_clustering_dict = cell_clustering(
        table_cluster,
        col_cluster,
        cell_cluster_cells_dict["X_temp"],
        cell_cluster_cells_dict["y_temp"],
        n_cell_clusters_per_col_cluster,
        n_cores,
        labels_per_cell_group
    )
    logging.debug("processing cluster %s ... done", str(col_cluster))
    return cell_cluster_cells_dict, cell_clustering_dict

def cell_cluster_sampling_labeling(cell_clustering_df, cell_cluster_cells_dict, 
                                   n_cores, classification_mode, tables_tuples_dict, 
                                   min_n_labels_per_cell_group, output_path, table_cluster, 
                                   col_cluster, manual_labeling_done
):
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
            original_data_keys_temp = cell_cluster_cells_dict["original_data_keys_temp"]
            datacells_uids = cell_cluster_cells_dict["datacells_uids"]

            if not manual_labeling_done:
                samples_dict, cell_clustering_df, n_user_labeled_cells = sampling(cell_clustering_df, X_temp, y_temp, value_temp, original_data_keys_temp, n_cores, tables_tuples_dict, min_n_labels_per_cell_group, datacells_uids)
                samples_dict_path = os.path.join(
                    output_path,
                    "samples_dict"
                    )
                os.makedirs(samples_dict_path, exist_ok=True)
                with open(os.path.join(samples_dict_path, "samples_dict_{}_{}.pkl".format(str(table_cluster), str(col_cluster))), "wb") as f:
                    pickle.dump(samples_dict, f)
                return None, None, None, None
            else:
                samples_dict_path = os.path.join(
                    output_path,
                    "samples_dict"
                    )
                with open(os.path.join(samples_dict_path, "samples_dict_{}_{}.pkl".format(str(table_cluster), str(col_cluster))), "rb") as f:
                    samples_dict = pickle.load(f)
                logging.info("Start labeling for cluster %s", str(cell_clustering_df["col_cluster"].values[0]))
                try:
                    samples_dict = labeling(samples_dict)
                except Exception as e:
                    logging.error("Error in labeling cluster %s", str(cell_clustering_df["col_cluster"].values[0]))
                    logging.error(e)
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
            if classification_mode == 0:
                X_train, y_train, X_test, y_test, y_cell_ids = get_train_test_sets(
                    X_temp, y_temp, samples_dict, cell_clustering_df
                )
                logging.info("start classification for cluster %s", str(cell_clustering_df["col_cluster"].values[0]))
                gbc, predicted = classify(X_train, y_train, X_test)
            elif classification_mode == 1:
                X_train, y_train, X_test, y_test, y_cell_ids, predicted = get_train_test_sets_per_col(
                    X_temp, y_temp, samples_dict, cell_clustering_df, cell_cluster_cells_dict["datacells_uids"]
                )
                
    except Exception as e:
        logging.error(
            "Error in cluster %s", str(cell_clustering_df["col_cluster"].values[0])
        )
        logging.error(e)

    cell_cluster_sampling_labeling_dict = {
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
    n_user_labeled_cells = len(X_labeled_by_user)
    return cell_cluster_sampling_labeling_dict, cell_clustering_df, samples_dict, n_user_labeled_cells


def cluster_column_group(col_groups_dir, df_n_labels, features_dict, labels_per_cell_group, file_name, n_cores):
    logging.info("Clustering column group: %s", file_name)
    table_clusters = []
    cell_cluster_cells_dict_all = {}
    cell_clustering_dict_all = {}
    col_clusters = []

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
            n_cell_groups = (math.floor(
                df_n_labels[
                    (df_n_labels["table_cluster"] == table_cluster)
                    & (df_n_labels["col_cluster"] == col_cluster)
                ]["n_labels"].values[0]/labels_per_cell_group)
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
                n_cores,
                labels_per_cell_group
            )
            cell_cluster_cells_dict_all[table_cluster][
                col_cluster
            ] = cell_cluster_cells_dict
            cell_clustering_dict_all[table_cluster][
                col_cluster
            ] = cell_clustering_dict

    logging.info("Clustering column group: %s ... done", file_name)

    return {"table_cluster": table_clusters, "col_clusters": col_clusters, "cell_cluster_cells_dict_all": cell_cluster_cells_dict_all, "cell_clustering_dict_all": cell_clustering_dict_all}
