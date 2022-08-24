from multiprocessing import freeze_support
import os

import pandas as pd

import app_logger
import check_results
import dataset_clustering
import extract_labels
import cols_grouping

import ed_twolevel_rahas_features
import pickle
from configparser import ConfigParser

import saving_results


def run_experiments(sandbox_path, output_path, exp_name, extract_labels_enabled, table_grouping_enabled,
                    column_grouping_enabled, labeling_budget, cell_feature_generator_enabled):
    labels_path = os.path.join(output_path, configs["DIRECTORIES"]["labels_filename"])
    if extract_labels_enabled:
        logger.info("Extracting labels started")
        labels_dict = extract_labels.generate_labels(sandbox_path, labels_path)
    else:
        with open(labels_path, 'rb') as file:
            labels_dict = pickle.load(file)
            logger.info("Labels loaded.")

    experiment_output_path = os.path.join(output_path, exp_name)

    if not os.path.exists(experiment_output_path):
        os.makedirs(experiment_output_path)
        logger.info("Experiment output directory is created.")

    # TODO: Fix this if-else
    table_grouping_output_path = os.path.join(output_path, configs["DIRECTORIES"]["table_grouping_output_filename"])
    # if table_grouping_enabled:
    logger.info("Table grouping started")
    table_grouping_output = dataset_clustering.cluster_datasets(sandbox_path, table_grouping_output_path,
                                                                configs["TABLE_GROUPING"][
                                                                    "auto_clustering_enabled"])
    # else:
    #     table_grouping_output = pd.read_csv(table_grouping_output_path)
    #     logger.info("Table grouping output loaded.")

    column_groups_path = os.path.join(experiment_output_path, configs["DIRECTORIES"]["column_groups_path"])
    if column_grouping_enabled:
        logger.info("Column grouping started")
        number_of_column_clusters = cols_grouping.col_folding(table_grouping_output, sandbox_path, labels_path,
                                                              column_groups_path,
                                                              configs["TABLE_GROUPING"]["auto_clustering_enabled"])
    else:
        number_of_column_clusters = cols_grouping.get_number_of_clusters(column_groups_path)
        print(number_of_column_clusters)

    if cell_feature_generator_enabled:
        features_dict = ed_twolevel_rahas_features.get_cells_features(sandbox_path, experiment_output_path)
        logger.info("Generating cell features started.")
    else:
        with open(os.path.join(experiment_output_path, configs["DIRECTORIES"]["cell_features_filename"]), 'rb') as file:
            features_dict = pickle.load(file)
            logger.info("Cell features loaded.")

    y_test, predicted, original_data_values = ed_twolevel_rahas_features.error_detector(column_groups_path,
                                                                                        experiment_output_path,
                                                                                        features_dict,
                                                                                        labeling_budget,
                                                                                        number_of_column_clusters,
                                                                                        "seq")
    tables_path = configs["RESULTS"]["tables_path"]
    results_path = os.path.join(experiment_output_path, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    saving_results.get_all_results(tables_path, results_path, original_data_values,
                                   y_test, predicted)
    check_results.get_all_results(output_path, results_path)


if __name__ == '__main__':

    logger = app_logger.get_logger()

    # App-Config Management
    configs = ConfigParser()
    configs.read("config.ini")

    sandbox_dir = configs["DIRECTORIES"]["sandbox_dir"]
    output_dir = configs["DIRECTORIES"]["output_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("Output directory is created.")

    freeze_support()
    exp_name = "BiWeeklyReport-Aug23"
    run_experiments(sandbox_dir, output_dir, exp_name, True, False, False, 20, False)
