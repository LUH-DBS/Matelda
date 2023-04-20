import logging
import os
import pickle

import pandas as pd

logger = logging.getLogger()


def extract_charset(col_groups_dir):
    clusters_char_set_dict = dict()
    table_charset_dict = dict()

    for file_name in os.listdir(col_groups_dir):
        if ".pickle" in file_name:
            file = open(os.path.join(col_groups_dir, file_name), 'rb')
            group_df = pickle.load(file)
            if not isinstance(group_df, pd.DataFrame):
                group_df = pd.DataFrame.from_dict(group_df, orient='index').T
            table_cluster = group_df['table_cluster'].values[0]
            file.close()
            clusters = set(group_df['column_cluster_label'].sort_values())
            for c_idx, cluster in enumerate(clusters):
                charset_val = []
                for idx, row in group_df[group_df['column_cluster_label'] == cluster].iterrows():
                    row = [str(val) for val in row['col_value']]
                    charset_val.extend(''.join(row))
                charset_val = set(charset_val)
                clusters_char_set_dict[(str(table_cluster), str(cluster))] = charset_val

            table_ids = set(group_df['table_id'].values)
            for table_id in table_ids:
                col_ids = set(group_df[group_df['table_id'] == table_id]['col_id'].values)
                for col_id in col_ids:
                    table_charset_dict[(str(table_id), str(col_id))] = clusters_char_set_dict[str(table_cluster), str(group_df[(group_df['table_id'] == table_id) & (group_df['col_id'] == col_id)]['column_cluster_label'].values[0])]
            logger.info("Charset dictionary generated.")
    return table_charset_dict