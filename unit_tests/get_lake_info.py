import os 
import html
import os.path

import numpy as np
import pandas as pd
import re
import csv

def read_changes_file(filename):
    data = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Split the first value based on '.'
            idx, attribute = row[0].split('.')
            dirty_value = row[1]
            clean_value = row[2]
            data.append({
                "idx": int(idx),
                "attribute": attribute,
                "dirty_value": dirty_value,
                "clean_value": clean_value
            })
    
    return pd.DataFrame(data)


def value_normalizer(value: str) -> str:
    """
    This method takes a value and minimally normalizes it. (Raha's value normalizer)
    """
    if value is not np.NAN:
        value = html.unescape(value)
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
    return value


def read_csv(path: str, low_memory: bool = False) -> pd.DataFrame:
    """
    This method reads a table from a csv file path, with pandas default null values and str data type
    Args:
        low_memory: whether to use low memory mode (bool), default False
        path: table path (str)

    Returns:
        pandas dataframe of the table
    """
    return pd.read_csv(path, sep=",", header="infer", low_memory=low_memory,keep_default_na=False, dtype=str).applymap(value_normalizer)

def get_info(sandbox_base_path, sandbox_name, dirty_file_names, clean_file_names):
    dataset_names= []
    for dir in os.listdir(os.path.join(sandbox_base_path, sandbox_name)):
        if dir != ".DS_Store" and dir != sandbox_name:
            dataset_names.append(dir)

    total_errors = 0
    total_n_cells = 0
    total_error_cells_injected = 0
    total_erroneous_rows = 0
    total_rows = 0
    total_cols = 0

    for dataset_name in dataset_names:
        table_path = os.path.join(sandbox_base_path, sandbox_name, dataset_name)
        dirty_df = read_csv(os.path.join(table_path, dirty_file_names))
        clean_df = read_csv(os.path.join(table_path, clean_file_names))
        if dirty_df.shape != clean_df.shape:
            print(dataset_name)
            print(dirty_df.shape)
            print(clean_df.shape)
        else:
            try:
                dirty_df.columns = clean_df.columns
                diff = dirty_df.compare(clean_df, keep_shape=True)
                self_diff = diff.xs('self', axis=1, level=1)
                other_diff = diff.xs('other', axis=1, level=1)

                label_df = ((self_diff != other_diff) & ~(self_diff.isna() & other_diff.isna())).astype(int)
                total_errors += label_df.sum().sum()
                total_erroneous_rows += (label_df == 1).any(axis=1).sum()
                total_n_cells += dirty_df.size
                total_rows += dirty_df.shape[0] 
                total_cols += dirty_df.shape[1]
                
            except:
                print(dataset_name)

    error_rate_cells = total_errors / total_n_cells
    error_rate_rows = total_erroneous_rows / total_rows
    avg_rows = total_rows / len(dataset_names)
    avg_cols = total_cols / len(dataset_names)

    lake_info_dict = {
        "sandbox_name": sandbox_name,
        "error_rate_cells": error_rate_cells,
        "error_rate_rows": error_rate_rows,
        "avg_rows": avg_rows,
        "avg_cols": avg_cols,
        "total_n_cells": total_n_cells,
        "total_errors": total_errors,
        "total_error_cells_injected": total_error_cells_injected,
        "total_erroneous_rows": total_erroneous_rows,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "total_datasets": len(dataset_names)
    }

    return lake_info_dict