import pandas as pd

from scourgify.col_grouping_module.extract_col_features import extract_col_features
from scourgify.load_module.load_data_cells import load_data_cells

results_path = "./test_mediate_files"
sand_box_dir = "./kaggle_sand"
dirty_file_name = "dirty_clean.csv"
clean_file_name = "clean.csv"

lake_dict = load_data_cells(sand_box_dir, dirty_file_name, save_results=True, results_path=results_path)
lake_dict_labeled = load_data_cells(sand_box_dir, dirty_file_name, save_results=False, results_path=results_path,
                                    clean_file_name=clean_file_name)
all_cols = []
char_set = set()
for table_id in lake_dict_labeled:
    for col_id in lake_dict_labeled[table_id]:
        col_values = []
        for cell_id in lake_dict_labeled[table_id][col_id]:
            value = lake_dict_labeled[table_id][col_id][cell_id][0]
            col_values.append(value)
            char_set.update(set(str(value)))
        all_cols.append(col_values)
extract_col_features(all_cols, char_set)