from collections import OrderedDict
import hashlib
import logging
import os
import pickle
import itertools
import subprocess
import time

from marshmallow_pipeline.utils.read_data import read_csv
from marshmallow_pipeline.cell_grouping_module.generate_raha_features import (
    generate_raha_features,
)


def get_cells_features(sandbox_path, output_path, table_char_set_dict, tables_dict, dirty_files_name, clean_files_name, save_mediate_res_on_disk, pool, raha_config):
    start_time = time.time()
    try:
        list_dirs_in_snd = os.listdir(sandbox_path)
        list_dirs_in_snd.sort()
        table_paths = [[table, sandbox_path, tables_dict[table], table_char_set_dict, dirty_files_name, clean_files_name] for table in list_dirs_in_snd if not table.startswith(".")]
        features_dict_list = []
        tables_tuples_list = []
        for table in list_dirs_in_snd:
             if not table.startswith("."):
                features_dict_tmp, tables_tuples_tmp = \
                    generate_cell_features(table, sandbox_path, tables_dict[table], table_char_set_dict, dirty_files_name, clean_files_name, pool, raha_config)
                features_dict_list.append(features_dict_tmp)
                tables_tuples_list.append(tables_tuples_tmp)
        features_dict = {k: v for d in features_dict_list for k, v in d.items()}
        tables_tuples_dict = {k: v for d in tables_tuples_list for k, v in d.items()}
        if save_mediate_res_on_disk:
            with open(os.path.join(output_path, "features.pickle"), "wb") as filehandler:
                pickle.dump(features_dict, filehandler)
            with open(os.path.join(output_path, "tables_tuples.pickle"), "wb") as filehandler:
                pickle.dump(tables_tuples_dict, filehandler)
    except Exception as e:
        logging.error(e)
    end_time = time.time()
    logging.info("Cell features generation time: " + str(end_time - start_time))
    return features_dict, tables_tuples_dict

def generate_cell_features(table, sandbox_path, table_file_name_santos, table_char_set_dict, dirty_files_name, clean_files_name, pool, raha_config):
    logging.info("Generate cell features; Table: %s", table)
    features_dict = {}
    table_tuples_dict = {}
    try:
        table_dirs_path = os.path.join(sandbox_path, table)

        dirty_df = read_csv(
            os.path.join(table_dirs_path, dirty_files_name), low_memory=False, data_type='str'
        )
        clean_df = read_csv(
            os.path.join(table_dirs_path, clean_files_name), low_memory=False, data_type='str'
        )

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
        table_tuples_dict[str(
                        hashlib.md5(
                            table_file_name_santos.encode()
                        ).hexdigest()
                    )] = {"header": None, "tuples": {}, "clean": {}}
        
        table_tuples_dict[str(
                        hashlib.md5(
                            table_file_name_santos.encode()
                        ).hexdigest()
                    )]["header"] = dirty_df.columns.tolist()
        logging.debug("Generate features ---- table: %s", table)
        t1 = time.time()
        col_features = generate_raha_features(
            sandbox_path, table, charsets, dirty_files_name, clean_files_name, pool, raha_config
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

        for row_idx in range(len(dirty_df)):
            table_tuples_dict[str(
                    hashlib.md5(
                        table_file_name_santos.encode()
                    ).hexdigest()
                )]["tuples"][row_idx] = dirty_df.iloc[row_idx].tolist()

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
        for row_idx in range(len(dirty_df)):
            table_tuples_dict[str(
                    hashlib.md5(
                        table_file_name_santos.encode()
                    ).hexdigest()
                )]["clean"][row_idx] = clean_df.iloc[row_idx].tolist()
        logging.debug("Table: %s", table)
    except Exception as e:
        logging.error(e)
        logging.error("Table: %s", table)

    return features_dict, table_tuples_dict

def check_spelling(words, words_set, checker='aspell'):

    # Prepare the input for the subprocess
    input_text = "\n".join(words_set)

    # Determine the command based on the checker
    if checker == 'aspell':
        spell_check_command = [checker, 'list']
    elif checker == 'hunspell':
        spell_check_command = [checker, '-l']
    else:
        raise ValueError("Invalid spell checker specified. Use 'aspell' or 'hunspell'.")

    # Run the spell checker as a subprocess
    result = subprocess.run(spell_check_command, input=input_text, text=True, capture_output=True)

    # The output contains misspelled words, one per line
    misspelled_words = set(result.stdout.splitlines())

    # Prepare the output list with 1s for misspelled words and 0s for correct ones
    spell_check_result = []
    for word in words:
        is_erroneous = 0
        if word in misspelled_words:
            is_erroneous = 1
        else:
            for subword in word.split(' '):
                if subword in misspelled_words:
                    is_erroneous = 1
                    break
        spell_check_result.append(is_erroneous)

    return spell_check_result


def get_cells_in_cluster(group_df, col_cluster, features_dict):
    original_data_keys_temp = [] # (table_id, col_id, cell_idx, cell_value)
    value_temp = []
    X_temp = []
    y_temp = []
    key_temp = []
    datacells_uids = {}
    current_local_cell_uid = 0
    all_table_cols = []
    cell_values_dict = OrderedDict()
    try:
        # appending colid features (onehot encoding)
        c_df = group_df[group_df["column_cluster_label"] == col_cluster]
        for _, row in c_df.iterrows():
            all_table_cols.append((row['table_id'], row['col_id']))
        for _, row in c_df.iterrows():
            table_col_id_features = OrderedDict()
            for table_col in all_table_cols:
                table_col_id_features[table_col] = 0
            table_col_id_features[(row['table_id'], row['col_id'])] = 1
            table_col_features_list = list(table_col_id_features.values())

            for cell_idx in range(len(row["col_value"])):
                original_data_keys_temp.append(
                    (
                        row["table_id"],
                        row["col_id"],
                        cell_idx,
                        row["col_value"][cell_idx],
                    )
                )

                value_temp.append(row["col_value"][cell_idx])
                complete_feature_vector = features_dict[
                    (row['table_id'], row['col_id'], cell_idx, 'og')
                    ].tolist()
                complete_feature_vector.extend(table_col_features_list)
                X_temp.append(complete_feature_vector)                
                y_temp.append(
                    features_dict[
                        (row["table_id"], row["col_id"], cell_idx, "gt")
                    ].tolist()
                )
                cell_values_dict[len(X_temp) - 1] = str(row["col_value"][cell_idx])
                key_temp.append((row["table_id"], row["col_id"], cell_idx))
                datacells_uids[
                    (
                        row["table_id"],
                        row["col_id"],
                        cell_idx,
                        row["col_value"][cell_idx],
                    )
                ] = current_local_cell_uid
                current_local_cell_uid += 1 

    except Exception as e:
        logging.error("Error in cluster {}".format(str(col_cluster)))
        logging.error(e)

    t0 = time.time()
    words_set = set(cell_values_dict.values())
    if 'nan' in words_set:
        words_set.remove('nan')
    aspell_spell_checker_output = check_spelling(cell_values_dict.values(), words_set, checker='aspell')
    t1 = time.time()
    logging.debug("Spell checker for this col group took %s seconds", str(t1 - t0))
    cell_values_dict_aspell = dict(zip(cell_values_dict.keys(), aspell_spell_checker_output))
    for fidx, feature_vector in enumerate(X_temp):
        feature_vector.append(cell_values_dict_aspell[fidx])
        X_temp[fidx] = feature_vector

    cell_cluster_cells_dict = {
        "col_cluster": col_cluster,
        "original_data_keys_temp": original_data_keys_temp,
        "value_temp": value_temp,
        "X_temp": X_temp,
        "y_temp": y_temp,
        "key_temp": key_temp,
        "datacells_uids": datacells_uids,
    }
    return cell_cluster_cells_dict