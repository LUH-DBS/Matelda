import logging
import math
import os

from col_grouping_module.extract_col_features import extract_col_features
from read_data import read_csv
import hashlib

logger = logging.getLogger()

def group_cols(path, table_grouping_dict, lake_base_path, labeling_budget, mediate_files_path):
    max_n_col_groups = math.floor(labeling_budget / len(table_grouping_dict) / 2)
    logger.info("group_cols")
    for table_group in table_grouping_dict:
        logger.info("table_group: {}".format(table_group))
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

        extract_col_features(table_group, cols, char_set, max_n_col_groups, mediate_files_path)

