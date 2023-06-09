"""This module contains the functions for grouping tables into clusters"""
import logging
import os
import pickle
import shutil
import subprocess

import networkx.algorithms.community as nx_comm

import marshmallow_pipeline.santos.codes.data_lake_processing_synthesized_kb
import marshmallow_pipeline.santos.codes.data_lake_processing_yago
import marshmallow_pipeline.santos.codes.query_santos
import marshmallow_pipeline.santos_fd.sortFDs_pickle_file_dict


def table_grouping(aggregated_lake_path: str, output_path: str) -> dict:
    """
    Group tables into clusters

    Args:
        graph_path (str): Path to the graph pickle file

    Returns:
        dict: Dictionary of table groups
    """
    logging.info("Table grouping")
    g_santos = run_santos(aggregated_lake_path=aggregated_lake_path)

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

    with open(os.path.join(output_path, "table_group_dict.pickle"), "wb+") as handle:
        pickle.dump(table_group_dict, handle)
    return table_group_dict


def run_santos(aggregated_lake_path: str):
    """
    Run santos on the sandbox.

    Args:
        aggregated_lake_path (str): Path to the sandbox

    Returns:
        _type_: _description_
    """
    logging.info("Preparing santos")
    santos_path = "marshmallow_pipeline/santos/benchmark/tus_benchmark"
    santos_lake_path = os.path.join(santos_path, "datalake")
    santos_query_path = os.path.join(santos_path, "query")

    logging.info("Symlinking sandbox to santos")
    os.makedirs(santos_path, exist_ok=True)
    shutil.rmtree(santos_lake_path, ignore_errors=True)
    shutil.rmtree(santos_query_path, ignore_errors=True)
    shutil.copytree(aggregated_lake_path, santos_lake_path, copy_function=os.link)
    shutil.copytree(aggregated_lake_path, santos_query_path, copy_function=os.link)

    logging.info("Santos run data_lake_processing_yago")
    # 1 == tus_benchmark
    marshmallow_pipeline.santos.codes.data_lake_processing_yago.main(1)

    logging.info("Creating functinal dependencies/ground truth for santos")
    datalake_files = [
        os.path.join(santos_lake_path, file)
        for file in os.listdir(santos_lake_path)
        if file.endswith(".csv")
    ]
    with open(
        "marshmallow_pipeline/santos_fd/tus_datalake_files.txt", "w+", encoding="utf-8"
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

    logging.info("Santos run sortFDs_pickle_file_dict")
    marshmallow_pipeline.santos_fd.sortFDs_pickle_file_dict.main()

    logging.info("Santos run data_lake_processing_synthesized_kb")
    # 1 == tus_benchmark
    marshmallow_pipeline.santos.codes.data_lake_processing_synthesized_kb.main(1)

    logging.info("Santos run query_santos")
    # 1 == tus_benchmark, 3 == full
    g_santos = marshmallow_pipeline.santos.codes.query_santos.main(1, 3)

    logging.info("Removing hardlinks")
    shutil.rmtree(santos_lake_path, ignore_errors=True)
    shutil.rmtree(santos_query_path, ignore_errors=True)

    logging.info("Santos finished")
    return g_santos
