"""
This module is the main module of the pipeline. It is responsible for
executing the pipeline steps and storing the results.
"""
import logging
import os
import pickle
from configparser import ConfigParser

import marshmallow_pipeline.utils.app_logger
from marshmallow_pipeline.error_detection import error_detector
from marshmallow_pipeline.grouping_columns import column_grouping
from marshmallow_pipeline.grouping_tables import table_grouping
from marshmallow_pipeline.saving_results import get_all_results
from marshmallow_pipeline.utils.loading_results import \
    loading_columns_grouping_results

if __name__ == "__main__":
    configs = ConfigParser()
    configs.read("config.ini")
    labeling_budget = int(configs["EXPERIMENTS"]["labeling_budget"])
    exp_name = configs["EXPERIMENTS"]["exp_name"]

    sandbox_path = configs["DIRECTORIES"]["sandbox_dir"]
    tables_path = os.path.join(sandbox_path, configs["DIRECTORIES"]["tables_dir"])

    experiment_output_path = os.path.join(
        configs["DIRECTORIES"]["output_dir"],
        "_"
        + exp_name
        + "_"
        + configs["DIRECTORIES"]["tables_dir"]
        + "_"
        + str(labeling_budget)
        + "_labels",
    )
    logs_dir = os.path.join(experiment_output_path, configs["DIRECTORIES"]["logs_dir"])
    results_path = os.path.join(
        experiment_output_path, configs["DIRECTORIES"]["results_dir"]
    )
    mediate_files_path = os.path.join(experiment_output_path, "mediate_files")
    aggregated_lake_path = os.path.join(
        experiment_output_path, configs["DIRECTORIES"]["aggregated_lake_path"]
    )

    os.makedirs(experiment_output_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(aggregated_lake_path, exist_ok=True)

    table_grouping_enabled = bool(int(configs["TABLE_GROUPING"]["tg_enabled"]))

    column_grouping_enabled = bool(int(configs["COLUMN_GROUPING"]["cg_enabled"]))
    min_num_labes_per_col_cluster = int(
        configs["COLUMN_GROUPING"]["min_num_labes_per_col_cluster"]
    )

    cell_feature_generator_enabled = bool(
        int(configs["CELL_GROUPING"]["cell_feature_generator_enabled"])
    )
    cell_clustering_alg = configs["CELL_GROUPING"]["cell_clustering_alg"]

    marshmallow_pipeline.utils.app_logger.setup_logging(logs_dir)
    logging.info("Starting the experiment")

    logging.info("Symlinking sandbox to aggregated_lake_path")
    tables_dict = {}
    for name in os.listdir(tables_path):
        curr_path = os.path.join(tables_path, name)
        if os.path.isdir(curr_path):
            dirty_csv_path = os.path.join(curr_path, "dirty_clean.csv")
            clean_csv_path = os.path.join(curr_path, "clean.csv")
            if os.path.isfile(dirty_csv_path):
                if os.path.exists(os.path.join(aggregated_lake_path, name + ".csv")):
                    os.remove(os.path.join(aggregated_lake_path, name + ".csv"))
                os.link(
                    dirty_csv_path, os.path.join(aggregated_lake_path, name + ".csv")
                )
                tables_dict[os.path.basename(curr_path)] = name + ".csv"

    if table_grouping_enabled:
        logging.info("Table grouping is enabled")
        logging.info("Executing the table grouping")
        table_grouping_dict = table_grouping(
            aggregated_lake_path, experiment_output_path
        )
    else:
        logging.info("Table grouping is disabled")
        logging.info("Loading the table grouping results...")
        with open(
            os.path.join(experiment_output_path, "table_group_dict.pickle"), "rb"
        ) as handle:
            table_grouping_dict = pickle.load(handle)

    if column_grouping_enabled:
        logging.info("Column grouping is enabled")
        logging.info("Executing the column grouping")
        column_grouping(
            aggregated_lake_path,
            table_grouping_dict,
            sandbox_path,
            labeling_budget,
            mediate_files_path,
        )
    else:
        logging.info("Column grouping is disabled")

    logging.info("Removing the symlinks")
    for name in os.listdir(tables_path):
        curr_path = os.path.join(tables_path, name)
        if os.path.isdir(curr_path):
            aggregated_lake_path_csv = os.path.join(aggregated_lake_path, name + ".csv")
            if os.path.exists(aggregated_lake_path_csv):
                os.remove(aggregated_lake_path_csv)

    logging.info("Loading the column grouping results")
    (
        number_of_col_clusters,
        cluster_sizes_dict,
        column_groups_df_path,
    ) = loading_columns_grouping_results(table_grouping_dict, mediate_files_path)

    logging.info("Starting error detection")
    # TODO: change output foldr of metanome
    # For fatemeh: no complete paths in code, Define a seed in config, with sets alls seeds in code(importance for reproducibility), document the code, no pickle! hard to read and securyt issue, if catching exception print exceptopn
    (
        y_test_all,
        y_local_cell_ids,
        predicted_all,
        y_labeled_by_user_all,
        unique_cells_local_index_collection,
        samples,
    ) = error_detector(
        cell_feature_generator_enabled,
        tables_path,
        column_groups_df_path,
        experiment_output_path,
        results_path,
        labeling_budget,
        number_of_col_clusters,
        cluster_sizes_dict,
        cell_clustering_alg,
        tables_dict,
        min_num_labes_per_col_cluster,
    )

    logging.info("Getting results")
    get_all_results(
        tables_dict,
        tables_path,
        results_path,
        y_test_all,
        y_local_cell_ids,
        predicted_all,
        y_labeled_by_user_all,
        unique_cells_local_index_collection,
        samples,
    )
