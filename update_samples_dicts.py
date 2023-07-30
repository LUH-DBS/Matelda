import pickle
import pandas as pd
import os

def update_dicts(samples_df_path, samples_dict_base_path):
    samples_df = pd.read_csv(samples_df_path)

    for i, row in samples_df.iterrows():
        path = os.path.join(samples_dict_base_path, f"samples_dict_{row['table_cluster']}_{row['col_cluster']}.pkl")
        with open(path, 'rb') as f:
            samples = pickle.load(f)
        samples['labels'][row['location_in_dict_i']][row['location_in_dict_j']] = row['label']
        with open(path, 'wb') as f:
            pickle.dump(samples, f)

samples_dict_base_path = "test-dirs/samples_dict"
samples_df_path = "test-dirs/sampling_df.csv"
update_dicts(samples_df_path, samples_dict_base_path)
