from configparser import ConfigParser
import os
import pickle

from cluster_tables import table_grouping
import app_logger


if __name__ == "__main__":
    configs = ConfigParser()
    configs.read("marshmallow_pipeline/config.ini")

    logs_dir = configs["DIRECTORIES"]["logs_dir"]
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
    c_graph_path = configs["TABLE_GROUPING"]["c_graph_path"]
    aggregated_lake_path = configs["COLUMN_GROUPING"]["aggregated_lake_path"]
    separated_lake_path = configs["COLUMN_GROUPING"]["separated_lake_path"]

    logger = app_logger.get_logger(logs_dir)
    logger.info("Starting the experiment")

    if table_grouping_enabled:
        logger.info("Table grouping is enabled")
        logger.info("Executing the table grouping")
        table_grouping_dict = table_grouping(c_graph_path)
    else:
        logger.info("Table grouping is disabled")
        logger.info("Loading the table grouping results...")
        with open(
            os.path.join(os.path.dirname(c_graph_path), "table_group_dict.pickle"), "rb"
        ) as handle:
            table_grouping_dict = pickle.load(handle)
