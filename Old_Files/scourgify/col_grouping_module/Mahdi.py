import os

import numpy as np

from scourgify.load_module.read_data import read_csv


def table_grouping_based_on_value_overlap(df_set):
    print("table_grouping_based_on_value_overlap")
    for df in df_set:
        for df_ in df_set:
            col_1 = df.columns
            col_2 = df_.columns
            # if col_1 == col_2:
            #     continue
            sum_overlap = 0
            counter_overlap = 0
            overlaps = []
            for c in col_1:
                max = 0
                for c_ in col_2:
                    val_1 = [str(val).lower() for val in df[c].values]
                    val_2 = [str(val).lower() for val in df_[c_].values]
                    overlap = len(set(val_1).intersection(set(val_2)))
                    if overlap > max:
                        max = overlap
                    overlaps.append(max)
                    sum_overlap += overlap
                    counter_overlap += 1
            if counter_overlap != 0:
                median = np.median(overlaps)
                print("df:", df)
                print("df_:", df_)
                print("median:", median)
            else:
                print("No Similarity!")

path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/scourgify/tests/kaggle_sand"
df_set = []
sandbox_children = os.listdir(path)
sandbox_children.sort()
table_id = 0
for child_name in sandbox_children:
    if not child_name.startswith("."):
        child_path = os.path.join(path, child_name)
        tables_dirs = os.listdir(child_path)
        tables_dirs.sort()
        for table in tables_dirs:
            table_path = os.path.join(child_path, table)
            if table.endswith("clean.csv"):
                clean_df = read_csv(os.path.join(table_path))
                df_set.append(clean_df)

table_grouping_based_on_value_overlap(df_set)
