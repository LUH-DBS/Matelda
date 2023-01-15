import math
import shutil
import time
import pandas as pd
from file_preprocessing import preprocess_headers, read_original_file, save_csv
from metanome_file_input import run_metanome
import create_bart_config
import numpy as np
import os
import subprocess
import random 
import pickle

def get_files_by_file_size(dirname, reverse=False):
    """ Return list of file paths in directory sorted by file size """

    # Get list of files
    filepaths = []
    for basename in os.listdir(dirname):
        filename = os.path.join(dirname, basename)
        if os.path.isfile(filename):
            filepaths.append(filename)

    # Re-populate list with filename, size tuples
    for i in range(len(filepaths)):
        filepaths[i] = (filepaths[i], os.path.getsize(filepaths[i]))

    # Sort list by file size
    # If reverse=True sort from largest to smallest
    # If reverse=False sort from smallest to largest
    filepaths.sort(key=lambda filename: filename[1], reverse=reverse)

    filenames = []
    # Re-populate list with just filenames
    for i in range(len(filepaths)):
        filenames.append(os.path.basename(filepaths[i][0]))

    return filenames

def find_det_dep(fd):
    determinant = fd['result']['determinant']['columnIdentifiers']
    dependant = fd['result']['dependant']['columnIdentifier']
    print()
    return determinant, dependant



def get_fd_list(fd_results):
    print("get_fd_list")
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
    config_file_path = create_bart_config.create_config_file(file_path, list(df.columns.values), outlier_error_cols, outlier_errors_percentage, typo_cols, typo_percentage, fd_ratio_dict, output_dir)
    val = subprocess.check_call("Sandbox_Generation_Extra_Apps_111222/BART/Bart_Engine/run.sh '%s'" % config_file_path, shell=True)

input_dir = "GitTables_1M_csv"
output_dir = "GitTables_1M_csv/processed"

for parent in os.listdir(input_dir):
    for par_sub_dir in os.listdir(os.path.join(input_dir, parent)):
        files = get_files_by_file_size(os.path.join(input_dir, parent, par_sub_dir))
        files_errors = dict()
        count = 0
        time_0 = time.time()
        for file in files:
            try:
                processed_file_path = os.path.join(output_dir, file.replace('.csv', ''))
                if not os.path.exists(processed_file_path):
                    os.makedirs(processed_file_path)
                    print(file + " is being processed.")
                    input_file_path = os.path.join(input_dir, parent, par_sub_dir, file)
                    df = read_original_file(input_file_path)
                    df = preprocess_headers(df)
                    df_name = save_csv(df, processed_file_path, file)
                    error_precentage = random.randint(1, 25)
                    files_errors[file] = error_precentage
                    make_it_dirty(error_precentage, os.path.join(processed_file_path, df_name), processed_file_path)
                    count += 1
                    print(file + " is done.")
                    if count % 10 == 0:
                        print(f'''{count} files processed.''' )
                        with open('files_error_percentages.pickle', 'wb') as handle:
                            pickle.dump(files_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(file, e)
time_1 = time.time()
print("********time*******:{} seconds".format(time_1-time_0))


    