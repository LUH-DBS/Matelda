"""
This module is the main module of the pipeline. It is responsible for
executing the pipeline steps and storing the results.
"""
import logging
import os
import pickle
from configparser import ConfigParser

import marshmallow_pipeline.utils.app_logger
from marshmallow_pipeline.grouping_tables import table_grouping
from marshmallow_pipeline.grouping_columns import column_grouping

if __name__ == "__main__":
    configs = ConfigParser()
    configs.read("config.ini")

    labeling_budget = int(configs["EXPERIMENTS"]["labeling_budget"])
    exp_name = configs["EXPERIMENTS"]["exp_name"]

    sandbox_path = configs["DIRECTORIES"]["sandbox_dir"]
    tables_path = os.path.join(sandbox_path, configs["DIRECTORIES"]["tables_dir"])

    experiment_output_path = os.path.join(configs["DIRECTORIES"]["output_dir"],  "_" + exp_name + "_" + configs["DIRECTORIES"]["tables_dir"] + "_" + str(labeling_budget)+ "_labels")
    logs_dir = os.path.join(experiment_output_path, configs["DIRECTORIES"]["logs_dir"])
    results_path = os.path.join(experiment_output_path, configs["DIRECTORIES"]["results_dir"])
    mediate_files_path = os.path.join(experiment_output_path, "mediate_files")
    aggregated_lake_path = os.path.join(experiment_output_path, configs["DIRECTORIES"]["aggregated_lake_path"])

    os.makedirs(experiment_output_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(aggregated_lake_path, exist_ok=True)

    table_grouping_enabled = bool(int(configs["TABLE_GROUPING"]["tg_enabled"]))

    column_grouping_enabled = bool(int(configs["COLUMN_GROUPING"]["cg_enabled"]))

    marshmallow_pipeline.utils.app_logger.setup_logging(logs_dir)
    logging.info("Starting the experiment")

    logging.info("Symlinking sandbox to aggregated_lake_path")
    for name in os.listdir(tables_path):
        curr_path = os.path.join(tables_path, name)
        if os.path.isdir(curr_path):
            dirty_csv_path = os.path.join(curr_path, "dirty_clean.csv")
            if os.path.isfile(dirty_csv_path):
                if os.path.exists(os.path.join(aggregated_lake_path, name + ".csv")):
                    os.remove(os.path.join(aggregated_lake_path, name + ".csv"))
                os.link(
                    dirty_csv_path, os.path.join(aggregated_lake_path, name + ".csv")
                )

    if table_grouping_enabled:
        logging.info("Table grouping is enabled")
        logging.info("Executing the table grouping")
        table_grouping_dict = table_grouping(aggregated_lake_path, experiment_output_path)
    else:
        logging.info("Table grouping is disabled")
        logging.info("Loading the table grouping results...")
        with open(os.path.join(experiment_output_path, "table_group_dict.pickle"),"rb") as handle:
            table_grouping_dict = pickle.load(handle)

    if column_grouping_enabled:
        logging.info("Column grouping is enabled")
        logging.info("Executing the column grouping")
        column_grouping(aggregated_lake_path, table_grouping_dict, sandbox_path, labeling_budget, mediate_files_path)
    else:
        logging.info("Column grouping is disabled")

    logging.info("Removing the symlinks")
    for name in os.listdir(tables_path):
        curr_path = os.path.join(tables_path, name)
        if os.path.isdir(curr_path):
            aggregated_lake_path_csv = os.path.join(aggregated_lake_path, name + ".csv")
            if os.path.exists(aggregated_lake_path_csv):
                os.remove(aggregated_lake_path_csv)
