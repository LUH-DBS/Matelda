import os

from scourgify.col_grouping_module.extract_col_features import extract_col_features
from scourgify.load_module.read_data import read_csv

path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/scourgify/datalake"
graph_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/scourgify/graph.gpickle"


def group_cols(path, table_grouping_dict):
    print("group_cols")
    for table_group in table_grouping_dict:
        print("table_group:", table_group)
        cols = []
        char_set = set()
        for table in table_grouping_dict[table_group]:
            df = read_csv(os.path.join(path, table))
            # convert the data frame to a string
            df_str = df.to_string()
            # create a set of unique characters
            char_set.update(set(df_str))
            for col in df.columns:
                cols.append(df[col].tolist())
        extract_col_features(table_group, cols, char_set)

