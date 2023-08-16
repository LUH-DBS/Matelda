import hashlib
import logging
import os
import pickle
import itertools
import time

from marshmallow_pipeline.utils.read_data import read_csv
from marshmallow_pipeline.cell_grouping_module.generate_raha_features import (
    generate_raha_features,
)


def get_cells_features(sandbox_path, output_path, table_char_set_dict, tables_dict, dirty_files_name, clean_files_name, save_mediate_res_on_disk, pool):
    start_time = time.time()
    try:
        list_dirs_in_snd = os.listdir(sandbox_path)
        list_dirs_in_snd.sort()
        table_paths = [[table, sandbox_path, tables_dict[table], table_char_set_dict, dirty_files_name, clean_files_name] for table in list_dirs_in_snd if not table.startswith(".")]
        features_dict_list = []
        for table in list_dirs_in_snd:
             if not table.startswith("."):
                features_dict_list.append(
                    generate_cell_features(table, sandbox_path, tables_dict[table], table_char_set_dict, dirty_files_name, clean_files_name, pool))
        features_dict = {k: v for d in features_dict_list for k, v in d.items()}
        if save_mediate_res_on_disk:
            with open(os.path.join(output_path, "features.pickle"), "wb") as filehandler:
                pickle.dump(features_dict, filehandler)
    except Exception as e:
        logging.error(e)
    end_time = time.time()
    logging.info("Cell features generation time: " + str(end_time - start_time))
    return features_dict

def generate_cell_features(table, sandbox_path, table_file_name_santos, table_char_set_dict, dirty_files_name, clean_files_name, pool):
    logging.info("Generate cell features; Table: %s", table)
    features_dict = {}
    try:
        table_dirs_path = os.path.join(sandbox_path, table)

        dirty_df = read_csv(
            os.path.join(table_dirs_path, dirty_files_name), low_memory=False, data_type='str'
        )
        clean_df = read_csv(
            os.path.join(table_dirs_path, clean_files_name), low_memory=False, data_type='str'
        )

        # TODO

        logging.debug("Generating features for table: %s", table)
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
        logging.debug("Generate features ---- table: %s", table)
        t1 = time.time()
        col_features = generate_raha_features(
            sandbox_path, table, charsets, dirty_files_name, clean_files_name, pool
        )
        t2 = time.time()
        logging.debug("Generate features ---- table: %s ---- took %s", table, str(t2-t1))
        logging.debug("Generate features done ---- table: %s", table)
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

        dirty_df.columns = clean_df.columns
        diff = dirty_df.compare(clean_df, keep_shape=True)
        self_diff = diff.xs('self', axis=1, level=1)
        other_diff = diff.xs('other', axis=1, level=1)
        # Custom comparison. True (or 1) only when values are different and not both NaN.
        label_df = ((self_diff != other_diff) & ~(self_diff.isna() & other_diff.isna())).astype(int)
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
        logging.debug("Table: %s", table)
    except Exception as e:
        logging.error(e)
        logging.error("Table: %s", table)

    return features_dict
