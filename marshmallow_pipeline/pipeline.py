from configparser import ConfigParser
from multiprocessing import freeze_support
import os
import pickle
import time

import pandas as pd
from error_detection import error_detector
from cluster_tables import table_grouping
from col_grouping_module.col_grouping import group_cols 
import saving_results
import app_logger


def main(labeling_budget):
    init_time = time.time()
    configs = ConfigParser()
    configs.read("/home/fatemeh/ED-Scale/marshmallow_pipeline/config-kaggle.ini")
    exp_name = configs["EXPERIMENTS"]["exp_name"]
    cell_feature_generator_enabled = bool(int(configs["CELL_GROUPING"]["cells_feature_generator_enabled"]))
    sandbox_path = configs["DIRECTORIES"]["sandbox_dir"]
    tables_path = configs["DIRECTORIES"]["tables_dir"]
    # labeling_budget = int(configs["EXPERIMENTS"]["labeling_budget"])

    experiment_output_path = os.path.join(configs["DIRECTORIES"]["output_dir"],  "_" + exp_name + "_" + str(labeling_budget)+ "_labels")
    logs_dir = os.path.join(experiment_output_path, "logs")
    results_path = os.path.join(experiment_output_path, "results")
    mediate_files_path = os.path.join(experiment_output_path, "mediate_files")

    if not os.path.exists(experiment_output_path):
        os.makedirs(experiment_output_path)
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    table_grouping_enabled = bool(int(configs["TABLE_GROUPING"]["tg_enabled"]))
    column_grouping_enabled = bool(int(configs["COLUMN_GROUPING"]["cg_enabled"]))
    santos_graph_path = configs["TABLE_GROUPING"]["santos_graph_path"]
    aggregated_lake_path = configs["COLUMN_GROUPING"]["aggregated_lake_path"]
    separated_lake_path = configs["COLUMN_GROUPING"]["separated_lake_path"]

    logger = app_logger.get_logger(logs_dir)
    logger.info("Starting the experiment")

    if table_grouping_enabled:
        logger.info("Table grouping is enabled")
        logger.info("Executing the table grouping")
        table_grouping_dict = table_grouping(santos_graph_path, mediate_files_path)
    else:
        logger.info("Table grouping is disabled")
        logger.info("Loading the table grouping results...")
        with open(os.path.join(os.path.dirname(mediate_files_path), 'table_group_dict.pickle'), 'rb') as handle:
            table_grouping_dict = pickle.load(handle)
    min_num_labes_per_col_cluster = int(configs["COLUMN_GROUPING"]["min_num_labes_per_col_cluster"])
    if column_grouping_enabled:
        logger.info("Column grouping is enabled")
        logger.info("Executing the column grouping")
        
        col_groups = group_cols(aggregated_lake_path, table_grouping_dict, separated_lake_path, labeling_budget, mediate_files_path, min_num_labes_per_col_cluster)

    n_table_groups = len(table_grouping_dict)
    number_of_col_clusters = {}
    col_groups = 0
    total_col_groups = 0
    cluster_sizes_dict = {"table_cluster": [], "col_cluster": [], "n_cells": []}
    column_groups_path = os.path.join(mediate_files_path, "col_grouping_res")
    column_groups_df_path = os.path.join(column_groups_path, "col_df_res")
    column_groups_cpc_path = os.path.join(column_groups_path, "cols_per_clu")

    with open(configs["CELL_GROUPING"]["tables_dict_path"], 'rb') as handle:
        tables_dict = pickle.load(handle)

    for i in range(n_table_groups):
        path = os.path.join(column_groups_cpc_path, 'cols_per_cluster_{}.pkl'.format(i))
        path_labels = os.path.join(column_groups_df_path, 'col_df_labels_cluster_{}.pickle'.format(i))
        dict_ = pickle.load(open(path, 'rb'))
        dict_labels = pickle.load(open(path_labels, 'rb'))
        labels_df = pd.DataFrame.from_dict(dict_labels, orient='index').T
        col_clusters = set(labels_df['column_cluster_label'])
        number_of_col_clusters[str(i)] = len(col_clusters)
        for cc in col_clusters:
            df = labels_df[labels_df['column_cluster_label'] == cc]
            n_cells = 0
            for idx, row in df.iterrows():
                n_cells += len(row['col_value'])
            cluster_sizes_dict["table_cluster"].append(i)
            cluster_sizes_dict["col_cluster"].append(cc)
            cluster_sizes_dict["n_cells"].append(n_cells)

    cell_clustering_alg= "km"

    print("starting error detection")

    y_test_all, y_local_cell_ids, predicted_all, y_labeled_by_user_all,\
        unique_cells_local_index_collection, samples = \
            error_detector(cell_feature_generator_enabled, sandbox_path, column_groups_df_path, experiment_output_path, results_path,\
                                                      labeling_budget, number_of_col_clusters, cluster_sizes_dict, cell_clustering_alg, tables_dict, min_num_labes_per_col_cluster)
    end_time = time.time()
    print("Total time: ", end_time - init_time)
    with open(os.path.join(results_path, "predicted.pickle"), "wb") as f:
        pickle.dump(predicted_all, f)
    saving_results.get_all_results(tables_dict, tables_path, results_path, y_test_all, y_local_cell_ids, predicted_all, y_labeled_by_user_all,\
    unique_cells_local_index_collection, samples)

# if __name__ == '__main__':
#     main(0)