import json
import logging
import os
import pickle

from scourgify.load_module.read_data import read_csv


def get_lake_cells_dict(table_id_dict: dict, sandbox_path: str, dirty_file_name: str, **kwargs) -> dict:
    """
    Get lake cells_dict dictionary from all tables in the sandbox.
    Args:
        table_id_dict:
        sandbox_path:
        dirty_file_name:
        kwargs:
            clean_file_name: clean file name (str),
            save_results: whether to save the results (bool),
            results_path: path to save the results (str)
    Returns:
        lake_cols_dict: lake cells_dict dictionary.
        lake cols dictionary format: {table_id: {col_id: {cell_id: cell_value}}}

    """
    if kwargs.get("clean_file_name", None):
        lake_cols_dict = {table_id_dict[table]: get_table_cells_dict(os.path.join(sandbox_path, table), dirty_file_name,
                                                                     clean_file_name=kwargs.get("clean_file_name", None)
                                                                     ) for table in table_id_dict}
    else:
        lake_cols_dict = {table_id_dict[table]: get_table_cells_dict(os.path.join(sandbox_path, table), dirty_file_name)
                          for table in table_id_dict}

    if kwargs.get("save_results", None):
        with open(os.path.join(kwargs.get("results_path", None), "lake_cols_dict.pickle"), 'wb') as filehandler:
            pickle.dump(lake_cols_dict, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
    return lake_cols_dict


def get_table_cells_dict(table_path: str, dirty_file_name: str, **kwargs) -> dict:
    """
    Get table cells_dict dictionary from a table.
    Args:
        table_path: table path.
        dirty_file_name: dirty file name. i.e. dirty.csv
        kwargs:
            clean_file_name: clean file name. i.e. clean.csv
            (we assumed that both clean and dirty file are in a same dir)

    Returns:
        cols_dict: table cols_dict dictionary.
        cols dictionary format: {col_id: {cell_id: (value, label)}}}

    """
    labels_dict = dict()
    dirty_df = read_csv(os.path.join(table_path, dirty_file_name))

    if kwargs.get("clean_file_name", None):
        clean_df = read_csv(os.path.join(table_path, kwargs.get("clean_file_name")))
        labels_df = load_labels(dirty_df, clean_df)
        labels_dict = labels_df.to_dict()

    # Convert the DataFrame to a dictionary
    df_dict = dirty_df.to_dict()

    # Create a nested dictionary for each column and cell
    cols_dict = {col_id: {index: (df_dict[col][index], labels_dict[col][index] if len(labels_dict) != 0 else None)
                          for index in df_dict[col]} for col_id, col in enumerate(df_dict)}

    return cols_dict


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


def load_labels(dirty_df, clean_df):
    """
    Load labels from dirty and clean dataframes.
    Args:
        dirty_df:   dirty dataframe.
        clean_df:   clean dataframe.

    Returns:
        labels_df:  labels dataframe.
    """
    labels_df = dirty_df.where(dirty_df.values != clean_df.values).notna() * 1
    return labels_df


def load_data_cells(sand_box_dir: str, dirty_file_name: str, **kwargs) -> dict:
    """
    Load lake cells dictionary from sandbox directory.
    Args:
        sand_box_dir: sandbox directory.
        dirty_file_name: dirty file name. i.e. dirty_clean.csv
        **kwargs:
            save_results: save results to disk (bool),
            results_path: path to save results (str)
            clean_file_name: clean file name. i.e. clean.csv
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
                                          results_path=kwargs.get("results_path", None),
                                          clean_file_name=kwargs.get("clean_file_name", None))
    return lake_cells_dict
