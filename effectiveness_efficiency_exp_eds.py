import os
import pandas
import configparser
import pipeline

def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def update_config(config, section, parameter, new_value):
    config.set(section, parameter, new_value)

def save_config(config, file_path):
    with open(file_path, 'w') as configfile:
        config.write(configfile)
new_temp_dir = f'/home/fatemeh/VLDB-Jan/ED-Scale-Dev/ED-Scale/temp_qpj'
if not os.path.exists(new_temp_dir):
    os.makedirs(new_temp_dir)
# Set the environment variable to your new location
# Note: Use 'TMPDIR' for Unix-like systems, 'TMP' for Windows
os.environ['TMPDIR'] = new_temp_dir  # Adjust the environment variable as needed

config_file_path = '/home/fatemeh/VLDB-Jan/ED-Scale-UL/Matelda/config.ini'
config = read_config(config_file_path)
datasets_info = pandas.read_csv("/home/fatemeh/VLDB-Jan/ED-Scale-UL/Matelda/datasets/datasets_info_all.csv")
dataset_names = ["DGov_NTR"]
labeling_budget_fractions = [0.10, 0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15, 20]
# labeling_budget_fractions = [0.10, 0.25, 0.5]
execution_times = 5
for exec in range(1, 5):
    for dataset_name in dataset_names:
        n_cols = datasets_info[datasets_info["sandbox_name"] == dataset_name]["total_cols"].values[0]
        update_config(config, 'DIRECTORIES', 'tables_dir', dataset_name)
        update_config(config, 'DIRECTORIES', 'output_dir', f"output_{dataset_name}_ul/output_{dataset_name}")
        for labeling_budget_fraction in labeling_budget_fractions:
            labeling_budget = round(n_cols*labeling_budget_fraction)
            update_config(config, 'EXPERIMENTS', 'labeling_budget', str(labeling_budget))
            save_config(config, config_file_path)
            pipeline.main(exec)