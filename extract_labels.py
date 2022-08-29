import logging
import os
import pickle
import pandas as pd 


# TODO: explain tables_ground_truth here

# TODO : add loger

def generate_labels(sandbox_path, output_path):
    sandbox_children = os.listdir(sandbox_path)
    tables_ground_truth = dict()
    table_id = 0
    for child_name in sandbox_children:
        child_path = os.path.join(sandbox_path, child_name)
        for table in os.listdir(child_path):
            try:
                table_path = os.path.join(child_path, table)
                dirty_df = pd.read_csv(table_path + "/dirty.csv", sep=",", header="infer", encoding="utf-8", dtype=str,
                                       keep_default_na=False, low_memory=False)
                clean_df = pd.read_csv(table_path + "/" + table + ".csv", sep=",", header="infer", encoding="utf-8",
                                       dtype=str, keep_default_na=False, low_memory=False)
                labels_df = dirty_df.where(dirty_df.values == clean_df.values).notna() * 1

                for column_id, column in enumerate(labels_df.columns.tolist()):
                    tables_ground_truth[(table_id, column_id)] = labels_df[column]
            except Exception as e:
                logging.exception("Error in generate_labels.", e)
            finally:
                table_id += 1

    with open(output_path, "wb") as filehandler:
        pickle.dump(tables_ground_truth, filehandler)

    return tables_ground_truth
