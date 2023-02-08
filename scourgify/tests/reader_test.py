import pandas as pd
from scourgify.load_module.read_data import read_csv

dirty_df_path = 'test_sand_box/Alcohol-use.g8_2014_0731_0900/dirty_clean.csv'
clean_df_path = 'test_sand_box/Alcohol-use.g8_2014_0731_0900/clean.csv'
errors_df_path = 'test_sand_box/Alcohol-use.g8_2014_0731_0900/clean_changes.csv'

dirty_df = read_csv(dirty_df_path, False)
clean_df = read_csv(clean_df_path, False)
errors_df = pd.read_csv(errors_df_path, header=None)

labels_df = dirty_df.where(dirty_df.values != clean_df.values).notna() * 1

assert sum(labels_df[labels_df == 1].count()) == errors_df.shape[0]
print("reader-test passed!")
