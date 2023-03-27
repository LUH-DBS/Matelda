import logging
import os
import pickle
import pandas as pd 


# TODO: explain tables_ground_truth here

# TODO : add loger

def generate_labels(sandbox_path, output_path):
    sandbox_children = os.listdir(sandbox_path)
    sandbox_children.sort()
    tables_ground_truth = dict()
    table_id = 0
    for child_name in sandbox_children:
        child_path = os.path.join(sandbox_path, child_name)
        tables_list = os.listdir(child_path)
        tables_list.sort()
        for table in tables_list:
            if not table.startswith("."):
                try:
                    table_path = os.path.join(child_path, table)
                    dirty_df = pd.read_csv(table_path + "/dirty_clean.csv", sep=",", header="infer", encoding="utf-8", dtype=str,
                                        keep_default_na=False, low_memory=False)
                    dirty_df = dirty_df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
                    dirty_df = dirty_df.replace('', 'NULL')
                    
                    clean_df = pd.read_csv(table_path + "/" + "clean.csv", sep=",", header="infer", encoding="utf-8",
                                        dtype=str, keep_default_na=False, low_memory=False)
                    clean_df = clean_df.replace('', 'NULL')
                    clean_df = clean_df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
                    labels_df = dirty_df.where(dirty_df.values != clean_df.values).notna() * 1

                    for column_id, column in enumerate(labels_df.columns.tolist()):
                        tables_ground_truth[(table_id, column_id)] = labels_df[column]
                except Exception as e:
                    logging.exception("Error in generate_labels.", e)
                finally:
                    table_id += 1

    with open(output_path, "wb") as filehandler:
        pickle.dump(tables_ground_truth, filehandler)

    return tables_ground_truth
