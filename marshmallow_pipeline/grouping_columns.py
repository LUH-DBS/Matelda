import hashlib
import logging
import math
import multiprocessing
import os
import pickle

from typing import Dict

from marshmallow_pipeline.column_grouping_module.col_grouping import (
    col_grouping,
)
from marshmallow_pipeline.utils.read_data import read_csv

def get_n_col_groups(table_grouping_dict, table_size_dict, labeling_budget):
    """
    This function calculates the number of column groups for each table group.  
    The number of column groups is calculated according to the following formula:
    min(n_cols_in_tg, floor(labeling_budget * n_cells_in_tg / count_all_cells / 2))
    where:
    n_cols_in_tg - the number of columns in the table group
    n_cells_in_tg - the number of cells in the table group
    count_all_cells - the number of cells in the entire dataset
    labeling_budget - the labeling budget

    Args:
        :param table_grouping_dict: A dictionary that maps between a table group and the tables in it.
        :param table_group_size_dict: A dictionary that maps between a table and the number of cells in it.
        :param labeling_budget: The labeling budget.
    
    Returns:
        :return: A dictionary that maps between a table group and the number of column groups, cols, and cells in it.

    """
    logging.info("Calculating the number of column groups for each table group")
    tg_stats = {}
    count_all_cells = 0
    count_all_cols = 0
    if len(table_grouping_dict) == 1:
        tg_stats[0] = {"n_cols": "Dummy Value", "n_cells": "Dummy Value", 
                       "max_n_col_groups": math.floor(labeling_budget / 2)}
        return tg_stats
    for table_group in table_grouping_dict:
        n_cols_in_tg = 0
        n_cells_in_tg = 0
        for table in table_grouping_dict[table_group]:
            n_cols_in_tg += table_size_dict[table][1]
            n_cells_in_tg += table_size_dict[table][0] * table_size_dict[table][1]
        count_all_cells += n_cells_in_tg
        count_all_cols += n_cols_in_tg
        tg_stats[table_group] = {"n_cols": n_cols_in_tg, "n_cells": n_cells_in_tg, "max_n_col_groups": 1}
    for table_group in table_grouping_dict:
        remained_labeling_budget = labeling_budget - 2 * sum(item['max_n_col_groups'] for item in tg_stats.values())
        max_n_col_groups = min(tg_stats[table_group]["n_cols"], 
            math.floor(remained_labeling_budget * tg_stats[table_group]["n_cols"] / count_all_cols / 2))# 2 is the minimum number of labels for each column group
        tg_stats[table_group]["max_n_col_groups"] += max_n_col_groups
    tg_stats = dict(sorted(tg_stats.items(), key = lambda item: item[1]['max_n_col_groups'], reverse=True))
    processed = 0
    i = 0
    while sum(item['max_n_col_groups'] for item in tg_stats.values()) < labeling_budget/2 and processed < len(table_grouping_dict):
        table_group = list(tg_stats.keys())[i]
        if tg_stats[table_group]["max_n_col_groups"] < tg_stats[table_group]["n_cols"]:
            tg_stats[table_group]["max_n_col_groups"] += 1
            processed = 0
        else:
            processed += 1
        if i < len(table_grouping_dict) - 1:
            i += 1 
        else:
            i = 0    
    total_n_col_groups = sum(val['max_n_col_groups'] for val in tg_stats.values())
    logging.info("Labeling Budget: %s, N Col Groups: %s", labeling_budget, total_n_col_groups)
    with open("tg_stats.pickle", "wb") as f:
        pickle.dump(tg_stats, f)
    return tg_stats

def column_grouping(
    path: str,
    table_grouping_dict: Dict,
    table_size_dict: Dict,
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
        :param table_size_dict: A dictionary that maps between a table group and the number of cells in it.
        :param lake_base_path: The path to the aggregated lake.
        :param labeling_budget: The labeling budget.
        :param mediate_files_path: The path to the mediate files.
        :param cg_enabled: A boolean that indicates whether the column grouping step is enabled.
        :param col_grouping_alg: The column grouping algorithm (km for minibatch kmeans or hac for hierarchical agglomerative clustering - default: hac).
        :param n_cores: The number of cores to use for parallelization.

    Returns:
        None
    """
    tg_stats = get_n_col_groups(table_grouping_dict, table_size_dict, labeling_budget)
    logging.info("Group columns")
    pool = multiprocessing.Pool()
    
    for table_group in table_grouping_dict:
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

        logging.debug("Apply for table_group: %s", table_group)
        pool.apply(
            col_grouping,
            args=(table_group, cols, char_set, tg_stats[table_group]["max_n_col_groups"], mediate_files_path, cg_enabled, col_grouping_alg, n_cores),
        )

    pool.close()
    pool.join()
