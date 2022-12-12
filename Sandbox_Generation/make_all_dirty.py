import math
import pandas as pd
from file_preprocessing import preprocess_headers, read_original_file, save_csv
from metanome_file_input import run_metanome
import create_bart_config
import numpy as np
import os
import subprocess
import random 


def find_det_dep(fd):
    determinant = fd['result']['determinant']['columnIdentifiers']
    dependant = fd['result']['dependant']['columnIdentifier']
    print()
    return determinant, dependant



def get_fd_list(fd_results):
    fd_list = []
    for fd in fd_results:
        determinant, dependant = find_det_dep(fd)
        if  len(determinant) == 1 and determinant != dependant \
                        and determinant != None and dependant != None \
                            and (dependant, determinant) not in fd_list:
            fd_list.append((determinant[0]['columnIdentifier'], dependant))
    return fd_list

def get_percentages(fd_list, error_percentage, outlier_error_cols, typo_cols):
    vio_gen_percentage, outlier_errors_percentage, typo_percentage = 0, 0, 0
    if len(fd_list) > 0: 
        vio_gen_percentage = math.floor(error_percentage/3) 
        if len(outlier_error_cols) > 0: 
            outlier_errors_percentage = math.floor((error_percentage - vio_gen_percentage)/2) 
            if len(typo_cols) > 0: 
                # FD, OE, SE
                typo_percentage = math.floor(error_percentage - vio_gen_percentage - outlier_errors_percentage) 
            else:
                # FD, OE
                vio_gen_percentage = math.floor(error_percentage/2)
                outlier_errors_percentage = math.floor(error_percentage/2)
        else:
            # FD, SE
            vio_gen_percentage = math.floor(error_percentage/2)
            typo_percentage = math.floor(error_percentage/2)
    else:
        if len(outlier_error_cols) > 0: 
            outlier_errors_percentage = math.floor(error_percentage/2) 
            if len(typo_cols) > 0: 
                # OE, SE
                typo_percentage = math.floor(error_percentage - outlier_errors_percentage) 
            else:
                # OE
                outlier_errors_percentage = error_percentage
        else:
            # SE
            typo_percentage = error_percentage
    return vio_gen_percentage, outlier_errors_percentage, typo_percentage

def set_fd_ratio(fd_list, vio_gen_percentage, num_table_records):

    num_fd_violations = num_table_records * (vio_gen_percentage/100)
    if num_fd_violations/2 < 1:
        return dict()

    fd_ratio_dict = dict()
    assinged_fds = 0
    i = 0

    while assinged_fds < num_fd_violations:
        if fd_list[i] in fd_ratio_dict:
            fd_ratio_dict[fd_list[i]] += (2 / num_table_records) * 100
        else: 
            fd_ratio_dict[fd_list[i]] = (2 / num_table_records) * 100
        assinged_fds += 1
        i += 1
        if i >= len(fd_list):
            i = 0
    return fd_ratio_dict

def make_it_dirty(error_percentage, file_path, output_dir):
    df = pd.read_csv(file_path)
    fd_results = run_metanome(file_path)
    fd_list = get_fd_list(fd_results)
    outlier_error_cols = list(df.select_dtypes(include=[np.number]).columns.values)
    typo_cols = list(df.select_dtypes(include=['object']).columns.values)
    vio_gen_percentage, outlier_errors_percentage, typo_percentage = get_percentages(fd_list, error_percentage, outlier_error_cols, typo_cols)  
    fd_ratio_dict = set_fd_ratio(fd_list, vio_gen_percentage, df.shape[0])
    create_bart_config.create_config_file(file_path, list(df.columns.values), outlier_error_cols, outlier_errors_percentage, typo_cols, typo_percentage, fd_ratio_dict, output_dir)

input_dir = "Sandbox_Generation/metanome_input_files"
output_dir = "Sandbox_Generation/metanome_input_files/processed"
config_files_path = "Sandbox_Generation/dirty_datasets/"

files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
count = 0
for file in files:
    print(file + " is being processed.")
    input_file_path = os.path.join(input_dir, file)
    df = read_original_file(input_file_path)
    df = preprocess_headers(df)
    save_csv(df,output_dir, file)
    config_file_path = make_it_dirty(random.randint(1, 25), os.path.join(input_dir, file), config_files_path)
    count += 1
    print(file + " is done.")
    if count // 10 == 0:
        print(f'''{count} files processed.''' )
    