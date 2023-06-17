import hashlib
import logging
import os
import pickle
import multiprocessing

from marshmallow_pipeline.utils.read_data import read_csv
from marshmallow_pipeline.cell_grouping_module.generate_raha_features import (
    generate_raha_features,
)


def get_cells_features(sandbox_path, output_path, table_char_set_dict, tables_dict):
    features_dict = {}
    list_dirs_in_snd = os.listdir(sandbox_path)
    list_dirs_in_snd.sort()
    table_paths = [[table, sandbox_path, tables_dict[table], table_char_set_dict] for table in list_dirs_in_snd if not table.startswith(".")]

    with multiprocessing.Pool() as pool:
        feature_dict_list = pool.starmap(generate_cell_features, table_paths)

    for feature_dict_tmp in feature_dict_list:
        features_dict.update(feature_dict_tmp)

    with open(os.path.join(output_path, "features.pickle"), "wb") as filehandler:
        pickle.dump(features_dict, filehandler)
    return features_dict

def generate_cell_features(table, sandbox_path, table_file_name_santos, table_char_set_dict):
    logging.info("************************ Table: %s", table)
    features_dict = {}
    try:
        table_dirs_path = os.path.join(sandbox_path, table)

        dirty_df = read_csv(
            os.path.join(table_dirs_path, "dirty_clean.csv"), low_memory=False
        )
        clean_df = read_csv(
            os.path.join(table_dirs_path + "/clean.csv"), low_memory=False
        )

        # TODO

        logging.info("Generating features for table: %s", table)
        charsets = {}
        for idx, _ in enumerate(dirty_df.columns):
            charsets[idx] = table_char_set_dict[
                (
                    str(
                        hashlib.md5(
                            table_file_name_santos.encode()
                        ).hexdigest()
                    ),
                    str(idx),
                )
            ]
        logging.info("Generate features ---- table: %s", table)
        col_features = generate_raha_features(
            sandbox_path, table, charsets
        )
        logging.info("Generate features done ---- table: %s", table)
        for col_idx, _ in enumerate(col_features):
            for row_idx, _ in enumerate(col_features[col_idx]):
                features_dict[
                    (
                        hashlib.md5(
                            table_file_name_santos.encode()
                        ).hexdigest(),
                        col_idx,
                        row_idx,
                        "og",
                    )
                ] = col_features[col_idx][row_idx]

        label_df = (
            dirty_df.where(
                dirty_df.astype(str).values != clean_df.astype(str).values
            ).notna()
            * 1
        )
        for col_idx, col_name in enumerate(label_df.columns):
            for row_idx in range(len(label_df[col_name])):
                features_dict[
                    (
                        hashlib.md5(
                            table_file_name_santos.encode()
                        ).hexdigest(),
                        col_idx,
                        row_idx,
                        "gt",
                    )
                ] = label_df[col_name][row_idx]
        logging.info("Table: %s", table)
    except Exception as e:
        logging.error(e)

    return features_dict
