import hashlib
import logging
import math
import multiprocessing
import os

from typing import Dict

from marshmallow_pipeline.column_grouping_module.col_grouping import (
    col_grouping,
)
from marshmallow_pipeline.utils.read_data import read_csv


def column_grouping(
    path: str,
    table_grouping_dict: Dict,
    table_group_size_dict: Dict,
    lake_base_path: str,
    labeling_budget: int,
    mediate_files_path: str,
    cg_enabled: bool,
    col_grouping_alg: str,
    n_cores: int,
) -> None:
    """
    This function is responsible for executing the column grouping step.

    Args:
        :param path: The path to the tables.
        :param table_grouping_dict: A dictionary that maps between a table group and the tables in it.
        :param table_group_size_dict: A dictionary that maps between a table group and the number of cells in it.
        :param lake_base_path: The path to the aggregated lake.
        :param labeling_budget: The labeling budget.
        :param mediate_files_path: The path to the mediate files.
        :param cg_enabled: A boolean that indicates whether the column grouping step is enabled.
        :param col_grouping_alg: The column grouping algorithm (km for minibatch kmeans or hac for hierarchical agglomerative clustering - default: hac).
        :param n_cores: The number of cores to use for parallelization.

    Returns:
        None
    """
    count_all_cells = sum(table_group_size_dict.values())
    logging.info("Group columns")
    pool = multiprocessing.Pool()

    for table_group in table_grouping_dict:
        max_n_col_groups = math.floor(labeling_budget * table_group_size_dict[table_group] / count_all_cells / 2) # 2 is the minimum number of labels for each column group
        logging.info("Table_group: %s", table_group)
        cols = {"col_value": [], "table_id": [], "table_path": [], "col_id": []}
        char_set = set()
        for table in table_grouping_dict[table_group]:
            df = read_csv(os.path.join(path, table))
            # convert the data frame to a string
            df_str = df.to_string()
            # create a set of unique characters
            char_set.update(set(df_str))
            for col_idx, col in enumerate(df.columns):
                cols["col_value"].append(df[col].tolist())
                cols["table_id"].append(hashlib.md5(table.encode()).hexdigest())
                cols["table_path"].append(os.path.join(lake_base_path, table))
                cols["col_id"].append(col_idx)

        pool.apply_async(
            col_grouping,
            args=(table_group, cols, char_set, max_n_col_groups, mediate_files_path, cg_enabled, col_grouping_alg, n_cores),
        )

    pool.close()
    pool.join()
