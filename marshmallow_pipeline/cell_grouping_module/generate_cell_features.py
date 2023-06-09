import hashlib
import logging
import os
import pickle

from marshmallow_pipeline.utils.read_data import read_csv
from marshmallow_pipeline.cell_grouping_module.generate_raha_features import generate_raha_features


def get_cells_features(sandbox_path, output_path, table_char_set_dict, tables_dict):
    features_dict = {}
    list_dirs_in_snd = os.listdir(sandbox_path)
    list_dirs_in_snd.sort()
    for parent in list_dirs_in_snd:
        table_dirs_path = os.path.join(sandbox_path, parent)
        table_dirs = os.listdir(table_dirs_path)
        table_dirs.sort()
        for table in table_dirs:
            if not table.startswith("."):
                logging.info("************************ Table: %s", table)
                try:
                    path = os.path.join(table_dirs_path, table)
                    table_file_name_santos = tables_dict[table]

                    dirty_df = read_csv(os.path.join(path, "dirty_clean.csv"), low_memory=False)
                    clean_df = read_csv(os.path.join(path + "/clean.csv"), low_memory=False)

                    # TODO
                    
                    logging.info("Generating features for table: %s", table)
                    charsets = {}
                    for idx, _ in enumerate(dirty_df.columns):
                        charsets[idx] = table_char_set_dict[(str(hashlib.md5(table_file_name_santos.encode()).hexdigest()), str(idx))]
                    logging.info("Generate features ---- table: %s", table)
                    col_features = generate_raha_features(table_dirs_path, table, charsets)
                    logging.info("Generate features done ---- table: %s", table)
                    for col_idx, _ in enumerate(col_features):
                        for row_idx, _ in enumerate(col_features[col_idx]):
                            features_dict[(hashlib.md5(table_file_name_santos.encode()).hexdigest(), col_idx, row_idx, 'og')] = col_features[col_idx][row_idx]
                            
                    
                    label_df = dirty_df.where(dirty_df.astype(str).values != clean_df.astype(str).values).notna() * 1
                    for col_idx, col_name in enumerate(label_df.columns):
                        for row_idx in range(len(label_df[col_name])):
                            features_dict[(hashlib.md5(table_file_name_santos.encode()).hexdigest(), col_idx, row_idx, 'gt')] = label_df[col_name][row_idx]
                    logging.info("Table: %s", table)
                except Exception as e:
                    logging.error(e)


    with open(os.path.join(output_path, "features.pickle"), "wb") as filehandler:
        pickle.dump(features_dict, filehandler)
    return features_dict
