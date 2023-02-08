import json
import logging
import os
import pickle
from collections import ChainMap

import numpy as np

from scourgify.load_module.read_data import read_csv


def get_lake_cells_dict(table_id_dict: dict, sandbox_path: str, dirty_file_name: str, **kwargs) -> dict:
    """
    Get lake cells_dict dictionary from all tables in the sandbox.
    Args:
        table_id_dict:
        sandbox_path:
        dirty_file_name:
        kwargs:
            save_results: whether to save the results (bool),
            results_path: path to save the results (str)
    Returns:
        lake_cells_dict: lake cells_dict dictionary.
        lake cells dictionary format: {hash((table_id, col, row)): (table_id, col, row, val)}

    """
    lake_cells_dict = [get_table_cells_dict(table_id_dict[table], os.path.join(sandbox_path, table), dirty_file_name)
                       for table in table_id_dict]
    lake_cells_dict = dict(ChainMap(*lake_cells_dict))
    if kwargs.get("save_results", None):
        with open(os.path.join(kwargs.get("results_path", None), "lake_cells_dict.pickle"), 'wb') as filehandler:
            pickle.dump(lake_cells_dict, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
    return lake_cells_dict


def get_table_cells_dict(table_id: int, table_path: str, dirty_file_name: str) -> dict:
    """
    Get table cells_dict dictionary from a table.
    Args:
        table_id: table id.
        table_path: table path.
        dirty_file_name: dirty file name. i.e. dirty_clean.csv

    Returns:
        cells_dict: table cells_dict dictionary.
        cells dictionary format: {hash((table_id, col, row)): (table_id, col, row, val)}

    """
    dirty_df = read_csv(os.path.join(table_path, dirty_file_name))
    values = dirty_df.to_numpy().flatten()
    row_indices, col_indices = np.indices(dirty_df.shape)
    cells_dict = {hash((table_id, col, row)): (table_id, col, row, val)
                  for row, col, val in zip(row_indices.flatten(), col_indices.flatten(), values)}
    return cells_dict


def get_table_id_dict(sand_box_dir: str, **kwargs) -> dict:
    """
    Get table id dictionary from sandbox directory.
    Args:
        sand_box_dir: sandbox directory.
        kwargs:
            save_results: save table id dictionary to disk,
            results_path: path to save table id dictionary.

    Returns:
        table_id_dict: table name to table id dictionary.
        table_id_dict format: {table_name: table_id}
    """
    table_names = []
    for dir_ in os.listdir(sand_box_dir):
        if not dir_.startswith(".") and dir_ != sand_box_dir:
            table_names.append(dir_)
    table_names.sort()
    table_id_dict = {table_name: table_id for table_id, table_name in enumerate(table_names)}
    if kwargs.get("save_results", None):
        with open(os.path.join(kwargs.get("results_path", None), "table_id_dict.json"), "w") as filehandler:
            json.dump(table_id_dict, filehandler)
    return table_id_dict


def load_data_cells(sand_box_dir: str, dirty_file_name: str, **kwargs) -> dict:
    """
    Load lake cells dictionary from sandbox directory.
    Args:
        sand_box_dir: sandbox directory.
        dirty_file_name: dirty file name. i.e. dirty_clean.csv
        **kwargs:
            save_results: save results to disk (bool),
            results_path: path to save results (str)
    Returns:
        lake_cells_dict: lake cells dictionary.
        lake cells dictionary format: {hash((table_id, col, row)): (table_id, col, row, val)}

    """
    logger = logging.getLogger()
    logger.info("Getting table id dictionary from sandbox directory: %s", sand_box_dir)
    table_id_dict = get_table_id_dict(sand_box_dir, save_results=kwargs.get("save_results", None),
                                      results_path=kwargs.get("results_path", None))
    lake_cells_dict = get_lake_cells_dict(table_id_dict, sand_box_dir, dirty_file_name,
                                          save_results=kwargs.get("save_results", None),
                                          results_path=kwargs.get("results_path", None))
    return lake_cells_dict
