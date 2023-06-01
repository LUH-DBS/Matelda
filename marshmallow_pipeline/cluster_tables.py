"""This module contains the functions for grouping tables into clusters"""
import logging
import os
import pickle
import subprocess

import networkx.algorithms.community as nx_comm

import marshmallow_pipeline.santos.codes.data_lake_processing_synthesized_kb
import marshmallow_pipeline.santos.codes.data_lake_processing_yago
import marshmallow_pipeline.santos.codes.query_santos
import marshmallow_pipeline.santos_fd.sortFDs_pickle_file_dict


def table_grouping(sandbox_path: str, output_path: str) -> dict:
    """
    Group tables into clusters

    Args:
        graph_path (str): Path to the graph pickle file

    Returns:
        dict: Dictionary of table groups
    """
    logging.info("Table grouping")
    g_santos = run_santos(sandbox_path=sandbox_path)

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

    with open(
        os.path.join(os.path.dirname(output_path), "table_group_dict.pickle"), "wb"
    ) as handle:
        pickle.dump(table_group_dict, handle)
    return table_group_dict


def run_santos(
    sandbox_path: str,
    santos_lake_path: str = "marshmallow_pipeline/santos/benchmark/tus_benchmark/datalake",
    santos_query_path: str = "marshmallow_pipeline/santos/benchmark/tus_benchmark/query",
):
    """
    Run santos on the sandbox.

    Args:
        sandbox_path (str): Path to the sandbox
        santos_lake_path (str, optional): _description_. 
            Defaults to "marshmallow_pipeline/santos/benchmark/tus_benchmark/datalake".
        santos_query_path (str, optional): _description_. 
            Defaults to "marshmallow_pipeline/santos/benchmark/tus_benchmark/query".

    Returns:
        _type_: _description_
    """
    logging.info("Preparing santos")

    logging.info("Symlinking sandbox to santos")
    os.makedirs(santos_lake_path, exist_ok=True)
    os.makedirs(santos_query_path, exist_ok=True)
    for name in os.listdir(sandbox_path):
        curr_path = os.path.join(sandbox_path, name)
        if os.path.isdir(curr_path):
            dirty_csv_path = os.path.join(curr_path, "dirty_clean.csv")
            if os.path.isfile(dirty_csv_path):

                if os.path.exists(os.path.join(santos_lake_path, name + ".csv")):
                    os.remove(os.path.join(santos_lake_path, name + ".csv"))
                os.link(
                    dirty_csv_path, os.path.join(santos_lake_path, name + ".csv")
                )

                if os.path.exists(os.path.join(santos_query_path, name + ".csv")):
                    os.remove(os.path.join(santos_query_path, name + ".csv"))
                os.link(
                    dirty_csv_path, os.path.join(santos_query_path, name + ".csv")
                )

    logging.info("Santos run data_lake_processing_yago")
    # 1 == tus_benchmark
    marshmallow_pipeline.santos.codes.data_lake_processing_yago.main(1)

    logging.info("Creating functinal dependencies/ground truth for santos")
    datalake_files = [os.path.join(santos_lake_path, file) for file in os.listdir(santos_lake_path) if file.endswith('.csv')]
    with open('marshmallow_pipeline/santos_fd/tus_datalake_files.txt', 'w+', encoding='utf-8') as file:
        file.write('\n'.join(datalake_files))

    process = subprocess.Popen(['bash', 'marshmallow_pipeline/santos_fd/runFiles.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with process.stdout:
        for line in iter(process.stdout.readline, b''): # b'\n'-separated lines
            logging.info("Santos FD: %s", line.decode("utf-8").strip())
    process.wait()
    
    logging.info("Santos run sortFDs_pickle_file_dict")
    marshmallow_pipeline.santos_fd.sortFDs_pickle_file_dict.main()

    logging.info("Santos run data_lake_processing_synthesized_kb")
    # 1 == tus_benchmark
    marshmallow_pipeline.santos.codes.data_lake_processing_synthesized_kb.main(1)

    logging.info("Santos run query_santos")
    # 1 == tus_benchmark, 3 == full
    g_santos = marshmallow_pipeline.santos.codes.query_santos.main(1, 3)

    logging.info("Removing hardlinks")
    os.makedirs(santos_lake_path, exist_ok=True)
    os.makedirs(santos_query_path, exist_ok=True)
    for name in os.listdir(sandbox_path):
        curr_path = os.path.join(sandbox_path, name)
        if os.path.isdir(curr_path):
            santos_lake_path_csv = os.path.join(santos_lake_path, name + ".csv")
            santos_query_path_csv = os.path.join(santos_query_path, name + ".csv")
            if os.path.exists(santos_lake_path_csv):
                os.remove(santos_lake_path_csv)
            if os.path.exists(santos_query_path_csv):
                os.remove(santos_query_path_csv)

    logging.info("Santos finished")
    return g_santos
