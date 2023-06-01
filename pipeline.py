"""
This module is the main module of the pipeline. It is responsible for
executing the pipeline steps and storing the results.
"""
import logging
import os
import pickle
from configparser import ConfigParser

import marshmallow_pipeline.app_logger
from marshmallow_pipeline.cluster_tables import table_grouping

if __name__ == "__main__":
    configs = ConfigParser()
    configs.read("config.ini")

    cell_feature_generator_enabled = bool(
        int(configs["CELL_GROUPING"]["cells_feature_generator_enabled"])
    )
    noise_extraction_enabled = bool(
        int(configs["CELL_GROUPING"]["noise_extraction_enabled"])
    )
    sandbox_path = configs["DIRECTORIES"]["sandbox_dir"]
    tables_path = configs["DIRECTORIES"]["tables_dir"]
    column_groups_path = configs["DIRECTORIES"]["column_groups_path"]
    column_groups_df_path = os.path.join(column_groups_path, "col_df_res")
    column_groups_cpc_path = os.path.join(column_groups_path, "cols_per_clu")
    experiment_output_path = configs["DIRECTORIES"]["output_dir"]
    results_path = configs["DIRECTORIES"]["results_dir"]
    logs_dir = configs["DIRECTORIES"]["logs_dir"]
    labeling_budget = int(configs["EXPERIMENTS"]["labeling_budget"])
    table_grouping_enabled = bool(int(configs["TABLE_GROUPING"]["tg_enabled"]))
    column_grouping_enabled = bool(int(configs["COLUMN_GROUPING"]["cg_enabled"]))
    aggregated_lake_path = configs["COLUMN_GROUPING"]["aggregated_lake_path"]
    separated_lake_path = configs["COLUMN_GROUPING"]["separated_lake_path"]

    marshmallow_pipeline.app_logger.setup_logging(logs_dir)
    logging.info("Starting the experiment")

    if table_grouping_enabled:
        logging.info("Table grouping is enabled")
        logging.info("Executing the table grouping")
        table_grouping_dict = table_grouping(tables_path, experiment_output_path)
    else:
        logging.info("Table grouping is disabled")
        logging.info("Loading the table grouping results...")
        with open(
            os.path.join(
                os.path.dirname(experiment_output_path), "table_group_dict.pickle"
            ),
            "rb",
        ) as handle:
            table_grouping_dict = pickle.load(handle)

    print(table_grouping_dict)
