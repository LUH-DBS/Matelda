"""This module contains the functions for grouping tables into clusters"""
import logging
import multiprocessing
import os
import pickle
import shutil
import subprocess
import time

from marshmallow_pipeline.table_grouping_module.table_grouping_bert import group_tables

import networkx.algorithms.community as nx_comm
import pandas as pd

import marshmallow_pipeline.santos.codes.data_lake_processing_synthesized_kb
import marshmallow_pipeline.santos.codes.data_lake_processing_yago
import marshmallow_pipeline.santos.codes.query_santos
import marshmallow_pipeline.santos_fd.sortFDs_pickle_file_dict

logger = logging.getLogger()

def table_grouping(aggregated_lake_path: str, output_path: str, table_grouping_method: str, save_mediate_res_on_disk: bool, pool:multiprocessing.Pool) -> dict:
    """
    Group tables into clusters

    Args:
        graph_path (str): Path to the graph pickle file
        output_path (str): Path to the output directory
        table_grouping_method (str): Table grouping method

    Returns:
        dict: Dictionary of table groups
    """
    logger.info("Table grouping")

    if table_grouping_method == "santos":
        g_santos, table_size_dict = run_santos(aggregated_lake_path=aggregated_lake_path, output_path=output_path)
        with open(os.path.join(output_path, "g_santos.pickle"), "wb+") as handle:
            pickle.dump(g_santos, handle)

        logging.info("Community detection")
        comp = nx_comm.louvain_communities(g_santos)

        logging.info("Creating table_group_dict")
        table_group_dict = {}
        table_group_dict_key = 0
        for community in comp:
            table_group_dict[table_group_dict_key] = []
            for table in community:
                table_group_dict[table_group_dict_key].append(table)
            table_group_dict_key += 1
    elif table_grouping_method == "bert":
        table_group_dict, table_size_dict = group_tables(aggregated_lake_path, batch_size=5, pool=pool)

    if save_mediate_res_on_disk:
        with open(os.path.join(output_path, "table_group_dict.pickle"), "wb+") as handle:
            pickle.dump(table_group_dict, handle)
        with open(os.path.join(output_path, "table_size_dict.pickle"), "wb+") as handle:
            pickle.dump(table_size_dict, handle)
    return table_group_dict, table_size_dict


def run_santos(aggregated_lake_path: str, output_path: str):
    """
    Run santos on the sandbox.

    Args:
        aggregated_lake_path (str): Path to the sandbox
        output_path (str): Path to the output directory

    Returns:
        nx.Graph: Santos graph
        _type_: _description_
    """
    logging.info("Preparing santos")
    santos_path = "marshmallow_pipeline/santos/benchmark/eds_benchmark"
    santos_lake_path = os.path.join(santos_path, "datalake")
    santos_query_path = os.path.join(santos_path, "query")

    logging.info("Symlinking sandbox to santos")
    os.makedirs(santos_path, exist_ok=True)
    os.makedirs(santos_lake_path, exist_ok=True)
    os.makedirs(santos_query_path, exist_ok=True)
    shutil.rmtree(santos_lake_path, ignore_errors=True)
    shutil.rmtree(santos_query_path, ignore_errors=True)
    # os.makedirs(santos_lake_path, exist_ok=True)
    # os.makedirs(santos_query_path, exist_ok=True)
    shutil.copytree(aggregated_lake_path, santos_lake_path, copy_function=os.link)
    shutil.copytree(aggregated_lake_path, santos_query_path, copy_function=os.link)
    # for file in os.listdir(aggregated_lake_path):
    #     if file.endswith(".csv"):
    #         df = read_csv(os.path.join(aggregated_lake_path, file)).sample(frac=0.1)
    #         df.to_csv(os.path.join(santos_lake_path, file), index=False)
    #         df.to_csv(os.path.join(santos_query_path, file), index=False)
            
    logging.info("Santos run data_lake_processing_yago")
    # 1 == eds_benchmark
    marshmallow_pipeline.santos.codes.data_lake_processing_yago.main(1)

    logging.info("Creating functinal dependencies/ground truth for santos")
    datalake_files = [
        os.path.join(santos_lake_path, file)
        for file in os.listdir(santos_lake_path)
        if file.endswith(".csv")
    ]
    with open(
        "marshmallow_pipeline/santos_fd/eds_datalake_files.txt", "w+", encoding="utf-8"
    ) as file:
        file.write("\n".join(datalake_files))

    process = subprocess.Popen(
        ["bash", "marshmallow_pipeline/santos_fd/runFiles.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with process.stdout:
        for line in iter(process.stdout.readline, b""):  # b'\n'-separated lines
            logging.info("Santos FD: %s", line.decode("utf-8").strip())
    process.wait()
    santos_fd_path = os.path.join(os.path.join(output_path, "santos_fds"), "results")
    if os.path.exists(santos_fd_path):
        shutil.rmtree(santos_fd_path, ignore_errors=True)

    os.makedirs(santos_fd_path)
    # List all files in the source directory
    files = os.listdir("results/")

    for file in files:
        # Move each file to destination Directory
        shutil.move(os.path.join("results/", file), os.path.join(santos_fd_path, file))
    # shutil.move("results/", santos_fd_path)

    logging.info("Santos run sortFDs_pickle_file_dict")
    marshmallow_pipeline.santos_fd.sortFDs_pickle_file_dict.main(santos_fd_path)

    logging.info("Santos run data_lake_processing_synthesized_kb")
    # 1 == tus_benchmark
    marshmallow_pipeline.santos.codes.data_lake_processing_synthesized_kb.main(1)

    logging.info("Santos run query_santos")
    # 1 == tus_benchmark, 3 == full
    g_santos, table_size_dict = marshmallow_pipeline.santos.codes.query_santos.main(1, 3)

    logging.info("Removing hardlinks")
    shutil.rmtree(santos_lake_path, ignore_errors=True)
    shutil.rmtree(santos_query_path, ignore_errors=True)

    logging.info("Santos finished")
    return g_santos, table_size_dict
