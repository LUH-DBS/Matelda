"""
This module is the main module of the pipeline. It is responsible for
executing the pipeline steps and storing the results.
"""
import logging
import multiprocessing
import os
import pickle
from configparser import ConfigParser
import time

import marshmallow_pipeline.utils.app_logger
from marshmallow_pipeline.error_detection import error_detector
from marshmallow_pipeline.grouping_columns import column_grouping
from marshmallow_pipeline.grouping_tables import table_grouping
from marshmallow_pipeline.saving_results_optimized import get_all_results
from marshmallow_pipeline.utils.loading_results import \
    loading_columns_grouping_results

def main(execution):
    configs = ConfigParser()
    configs.read("/home/fatemeh/ED-Scale-Oct/ED-Scale/config.ini")
    # labeling_budget = int(configs["EXPERIMENTS"]["labeling_budget"])
    exp_name = configs["EXPERIMENTS"]["exp_name"]
    n_cores = int(configs["EXPERIMENTS"]["n_cores"])
    save_mediate_res_on_disk = bool(int(configs["EXPERIMENTS"]["save_mediate_res_on_disk"]))

    sandbox_path = configs["DIRECTORIES"]["sandbox_dir"]
    tables_path = os.path.join(sandbox_path, configs["DIRECTORIES"]["tables_dir"])

    raha_config = {}
    raha_config['save_results'] = bool(int(configs["RAHA"]['save_results']))
    raha_config['strategy_filtering'] = bool(int(configs["RAHA"]['strategy_filtering']))
    raha_config['error_detection_algorithms'] = configs["RAHA"]['error_detection_algorithms'].split(', ')

    experiment_output_path = os.path.join(
        configs["DIRECTORIES"]["output_dir"] + f"_{execution}",
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
    os.makedirs(logs_dir + f"_{exp_name}", exist_ok=True)
    os.makedirs(aggregated_lake_path, exist_ok=True)
    os.makedirs(mediate_files_path, exist_ok=True)

    table_grouping_enabled = bool(int(configs["TABLE_GROUPING"]["tg_enabled"]))
    table_grouping_res_available = bool(int(configs["TABLE_GROUPING"]["tg_res_available"]))
    table_grouping_method = configs["TABLE_GROUPING"]["tg_method"]

    column_grouping_enabled = bool(int(configs["COLUMN_GROUPING"]["cg_enabled"]))
    column_grouping_res_available = bool(int(configs["COLUMN_GROUPING"]["cg_res_available"]))
    column_grouping_alg = configs["COLUMN_GROUPING"]["cg_clustering_alg"]
    min_num_labes_per_col_cluster = int(
        configs["COLUMN_GROUPING"]["min_num_labes_per_col_cluster"]
    )

    cell_feature_generator_enabled = bool(
        int(configs["CELL_GROUPING"]["cell_feature_generator_enabled"])
    )
    cell_clustering_alg = configs["CELL_GROUPING"]["cell_clustering_alg"]
    cell_clustering_res_available = bool(int(configs["CELL_GROUPING"]["cell_clustering_res_available"]))
    classification_mode = int(configs["CELL_GROUPING"]["classification_mode"])
    if classification_mode == 3:
        nearest_neighbours_percentage = float(configs["CELL_GROUPING"]["nearest_neighbours_percentage"])
    else:
        nearest_neighbours_percentage = -1
    labeling_method = int(configs["CELL_GROUPING"]["labeling_method"])
    if labeling_method == 2:
        llm_labels_per_cell_group = int(configs["CELL_GROUPING"]["llm_labels_per_cell_group"])
    else: 
        llm_labels_per_cell_group = -1
    min_n_labels_per_cell_group = int(configs["CELL_GROUPING"]["labels_per_cell_group"])

    dirty_files_name = configs["DIRECTORIES"]["dirty_files_name"]
    clean_files_name = configs["DIRECTORIES"]["clean_files_name"]

    marshmallow_pipeline.utils.app_logger.setup_logging(logs_dir + f"_{exp_name}")
    logging.info("Starting the experiment")
    time_start = time.time()
    pool = multiprocessing.Pool(n_cores)

    logging.info("Symlinking sandbox to aggregated_lake_path")
    tables_dict = {}
    for name in os.listdir(tables_path):
        curr_path = os.path.join(tables_path, name)
        if os.path.isdir(curr_path):
            dirty_csv_path = os.path.join(curr_path, dirty_files_name)
            clean_csv_path = os.path.join(curr_path, clean_files_name)
            if os.path.isfile(dirty_csv_path):
                if os.path.exists(os.path.join(aggregated_lake_path, name + ".csv")):
                    os.remove(os.path.join(aggregated_lake_path, name + ".csv"))
                os.link(
                    dirty_csv_path, os.path.join(aggregated_lake_path, name + ".csv")
                )
                tables_dict[os.path.basename(curr_path)] = name + ".csv"

    if save_mediate_res_on_disk:
        with open(os.path.join(experiment_output_path, "tables_dict.pickle"), "wb+") as handle:
            pickle.dump(tables_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Table grouping
    if table_grouping_enabled:
        if not table_grouping_res_available:
            logging.info("Table grouping is enabled")
            logging.info("Table grouping results are not available")
            logging.info("Executing the table grouping")
            before_tg = time.time()
            logging.debug("Thread pool: " + str(before_tg - time_start))
            table_g_start = time.time()
            table_grouping_dict, table_size_dict = table_grouping(
                aggregated_lake_path, experiment_output_path, table_grouping_method, save_mediate_res_on_disk, pool
            )

            table_g_time = time.time() - table_g_start
            logging.info("Table grouping is done")
            logging.info("Table grouping time: " + str(table_g_time))
        else:
            logging.info("Table grouping results are available")
            logging.info("Loading the table grouping results...")
            with open(
                os.path.join(experiment_output_path, "table_group_dict.pickle"), "rb"
            ) as handle:
                table_grouping_dict = pickle.load(handle)
            with open(
                os.path.join(experiment_output_path, "table_size_dict.pickle"),
                "rb",
            ) as handle:
                table_size_dict = pickle.load(handle)
    else:
        logging.info("Table grouping is disabled")
        table_grouping_dict = {0:[]}
        table_size_dict = {0:-1}
        tables_list = tables_dict.values()
        for table in tables_list:
            table_grouping_dict[0].append(table)
        if save_mediate_res_on_disk:
            with open(os.path.join(experiment_output_path, "table_group_dict.pickle"), "wb+") as handle:
                pickle.dump(table_grouping_dict, handle)

    logging.info("Table grouping is done")
    logging.info("I need at least 2 labeled cells per table group to work at all and at least 2 * 6 labeled cells per table group to work effectively! Thant means you need to label {} cells if you want reasonable (!) results:".format(2*6*len(table_grouping_dict)))
    print("I need at least 2 labeled cells per table group to work at all and at least 2 * 6 labeled cells per table group to work effectively! Thant means you need to label {} cells if you want reasonable (!) results:".format(2*6*len(table_grouping_dict)))

    if labeling_budget < 2* 6 *len(table_grouping_dict):
        labeling_budget = 2*6*len(table_grouping_dict)
    # Column grouping
    if not column_grouping_res_available:
        logging.info("Column grouping results are not available")
        logging.info("Executing the column grouping")
        column_grouping(
            aggregated_lake_path,
            table_grouping_dict,
            table_size_dict,
            sandbox_path,
            labeling_budget,
            min_n_labels_per_cell_group,
            mediate_files_path,
            column_grouping_enabled,
            column_grouping_alg,
            n_cores,
            pool
    )
    else:
        logging.info("Column grouping results are available - loading from disk")
        

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
    (
        y_test_all,
        y_local_cell_ids,
        predicted_all,
        y_labeled_by_user_all,
        unique_cells_local_index_collection,
        samples, global_n_userl_labels, global_n_model_labels
    ) = error_detector(
        cell_feature_generator_enabled,
        tables_path,
        column_groups_df_path,
        experiment_output_path,
        results_path,
        labeling_budget,
        min_n_labels_per_cell_group,
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
    )

    time_end = time.time()
    print(time_end - time_start)
    # logging.CRITICAL(f"The experiment took {str(time_end - time_start)} seconds")
    with open(os.path.join(results_path, "time.txt"), "w") as file:
        file.write(str(time_end - time_start))
    

    logging.info("Saving the results")
    final_results_path = os.path.join(results_path, "final_results")
    os.makedirs(final_results_path, exist_ok=True)
    with open(os.path.join(final_results_path, "tables_dict.pickle"), "wb+") as handle:
        pickle.dump(tables_dict, handle)
    with open(os.path.join(final_results_path, "y_test_all.pickle"), "wb+") as handle:
        pickle.dump(y_test_all, handle)
    with open(os.path.join(final_results_path, "y_local_cell_ids.pickle"), "wb+") as handle:
        pickle.dump(y_local_cell_ids, handle)
    with open(os.path.join(final_results_path, "predicted_all.pickle"), "wb+") as handle:
        pickle.dump(predicted_all, handle)
    with open(os.path.join(final_results_path, "y_labeled_by_user_all.pickle"), "wb+") as handle:
        pickle.dump(y_labeled_by_user_all, handle)
    with open(os.path.join(final_results_path, "unique_cells_local_index_collection.pickle"), "wb+") as handle:
        pickle.dump(unique_cells_local_index_collection, handle)
    with open(os.path.join(final_results_path, "samples.pickle"), "wb+") as handle:
        pickle.dump(samples, handle)

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
        dirty_files_name,
        clean_files_name
    )
    
    logging.info(f"Number of user labeled cells: {global_n_userl_labels}, Number of model labeled cells {global_n_model_labels}")

# if __name__ == "__main__":
#     main(100, 1)
