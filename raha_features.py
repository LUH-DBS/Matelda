import functools
import hashlib
import itertools
import json
import multiprocessing
import os
import pickle
import random
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import raha
import sklearn
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import Row


def generate_raha_features_pyspark(
    csv_paths_df: DataFrame,
    grouped_column_groups_df: DataFrame,
    column_grouping_df: DataFrame,
    raha_features_path: str,
    cell_feature_generator_enabled: int,
) -> None:
    """_summary_

    Args:
        csv_paths_df (DataFrame): _description_
        grouped_column_groups_df (DataFrame): _description_
        raha_features_path (str): _description_
        cell_feature_generator_enabled (int): _description_

    Returns:
        DataFrame: _description_
    """
    if cell_feature_generator_enabled == 1:
        spark = SparkSession.getActiveSession()
        log4j_logger = spark._jvm.org.apache.log4j
        logger = log4j_logger.LogManager.getLogger(__name__)

        logger.warn("Creating Raha features")
        logger.warn("Writing Raha features to file")

        csv_paths_df.join(grouped_column_groups_df, "table_id", "inner").rdd.flatMap(
            generate_raha_features
        ).toDF(["table_id", "column_id", "row_id", "features"]).join(
            column_grouping_df, ["table_id", "column_id"], "inner"
        ).write.parquet(
            raha_features_path, mode="overwrite"
        )


def generate_raha_features(row: Row) -> List[Tuple[int, int, int, Any]]:
    """_summary_

    Args:
        row (Row): _description_

    Returns:
        List[Tuple[int, int, int, Any]]: _description_
    """
    detect = raha.detection.Detection()
    detect.SAVE_RESULTS = False
    detect.VERBOSE = False

    dataset_dictionary = {
        "name": row.table_name,
        "path": row.dirty_path,
        "clean_path": row.clean_path,
    }

    d = detect.initialize_dataset(dataset_dictionary)
    d.SAVE_RESULTS = False
    d.VERBOSE = False
    d.ERROR_DETECTION_ALGORITHMS = ["OD", "PVD", "RVD", "TFIDF"]

    run_strategies(detect, d, row.characters)
    generate_features(detect, d)

    feature_list = []

    for col_idx in range(len(d.column_features)):
        for row_idx in range(len(d.column_features[col_idx])):
            # print(len(d.column_features[col_idx][row_idx]))
            feature_list.append(
                (
                    row.table_id,
                    col_idx,
                    row_idx,
                    Vectors.dense(
                        np.append(
                            d.column_features[col_idx][row_idx],
                            row.table_id,  # TODO: remove table_id?
                        )
                    ),
                )
            )

    return feature_list


# TODO: Mark this code part as adapted code from raha (Apache License 2.0 (4b) requires this )


def run_strategies(
    self: raha.detection.Detection, d: raha.dataset.Dataset, char_set: List[List[str]]
) -> None:
    """This method runs (all or the promising) strategies.

    Args:
        self (raha.detection.Detection): _description_
        d (raha.dataset.Dataset): _description_
        char_set (List[List[str]]): _description_
    """
    sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
    if not self.STRATEGY_FILTERING:
        if os.path.exists(sp_folder_path):
            sys.stderr.write(
                "I just load strategies' results as they have already been run on the dataset!\n"
            )
            strategy_profiles_list = [
                pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                for strategy_file in os.listdir(sp_folder_path)
                if not strategy_file.startswith(".")
            ]
        else:
            if self.SAVE_RESULTS:
                os.mkdir(sp_folder_path)
            algorithm_and_configurations = []
            for algorithm_name in self.ERROR_DETECTION_ALGORITHMS:
                if algorithm_name == "OD":  # 34
                    configuration_list = [
                        list(a)
                        for a in list(
                            itertools.product(
                                ["histogram"],
                                ["0.1", "0.3", "0.5", "0.7", "0.9"],
                                ["0.1", "0.3", "0.5", "0.7", "0.9"],
                            )
                        )
                        + list(
                            itertools.product(
                                ["gaussian"],
                                [
                                    "1.0",
                                    "1.3",
                                    "1.5",
                                    "1.7",
                                    "2.0",
                                    "2.3",
                                    "2.5",
                                    "2.7",
                                    "3.0",
                                ],
                            )
                        )
                    ]
                    algorithm_and_configurations.extend(
                        [
                            [d, algorithm_name, configuration]
                            for configuration in configuration_list
                        ]
                    )
                elif algorithm_name == "PVD":
                    for j, attribute in enumerate(d.dataframe.columns):
                        characters_dictionary = {ch: 1 for ch in char_set[j]}
                        for ch in characters_dictionary:
                            configuration_list.append([attribute, ch])
                    algorithm_and_configurations.extend(
                        [
                            [d, algorithm_name, configuration]
                            for configuration in configuration_list
                        ]
                    )

                elif algorithm_name == "RVD":
                    configuration_list = []
                    configuration_list.append("FCD")
                    configuration_list.append("RND")
                    configuration_list.append("LND")
                    algorithm_and_configurations.extend(
                        [
                            [d, algorithm_name, configuration]
                            for configuration in configuration_list
                        ]
                    )

            random.shuffle(algorithm_and_configurations)

            pool = multiprocessing.Pool()
            _strategy_runner_process_ = functools.partial(_strategy_runner_process, d)
            strategy_profiles_list = pool.map(
                _strategy_runner_process_, algorithm_and_configurations
            )
            pool.close()
            pool.join()
    else:
        for dd in self.HISTORICAL_DATASETS + [d.dictionary]:
            raha.utilities.dataset_profiler(dd)
            raha.utilities.evaluation_profiler(dd)
        strategy_profiles_list = (
            raha.utilities.get_selected_strategies_via_historical_data(
                d.dictionary, self.HISTORICAL_DATASETS
            )
        )
    d.strategy_profiles = strategy_profiles_list
    if self.VERBOSE:
        print("{} strategy profiles are collected.".format(len(d.strategy_profiles)))


def generate_features(
    self: raha.detection.Detection, d: raha.dataset.Dataset
) -> List[np.ndarray]:
    """This method generates features.

    Args:
        self (raha.detection.Detection): _description_
        d (raha.dataset.Dataset): _description_

    Returns:
        List[np.ndarray]: _description_
    """
    columns_features_list = []
    col_name_features = np.asarray([hash(col_name) for col_name in d.dataframe.columns])
    for j in range(d.dataframe.shape[1]):
        feature_vectors = np.zeros((d.dataframe.shape[0], len(d.strategy_profiles)))
        for strategy_index, strategy_profile in enumerate(d.strategy_profiles):
            strategy_name = json.loads(strategy_profile["name"])[0]
            if strategy_name in self.ERROR_DETECTION_ALGORITHMS:
                for cell in strategy_profile["output"]:
                    if cell[1] == j:
                        feature_vectors[cell[0], strategy_index] = 1.0
        if "TFIDF" in self.ERROR_DETECTION_ALGORITHMS:
            vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
                min_df=1, stop_words="english"
            )
            corpus = d.dataframe.iloc[:, j]
            try:
                tfidf_features = vectorizer.fit_transform(corpus)
                feature_vectors = np.column_stack(
                    (feature_vectors, np.array(tfidf_features.todense()))
                )
            except Exception:
                pass

        if self.VERBOSE:
            print(
                "{} Features are generated for column {}.".format(
                    feature_vectors.shape[1], j
                )
            )
        # Adding headers hash

        feature_vectors = np.hstack(
            (
                feature_vectors,
                np.full((feature_vectors.shape[0], 1), col_name_features[j]),
            )
        )
        columns_features_list.append(feature_vectors)

    d.column_features = columns_features_list


def _strategy_runner_process(self: raha.detection.Detection, args: List[Any]) -> Dict:
    """This method runs an error detection strategy in a parallel process.

    Args:
        self (raha.detection.Detection): _description_
        args (List[Any]): _description_

    Returns:
        Dict: _description_
    """
    d, algorithm, configuration = args
    start_time = time.time()
    strategy_name = json.dumps([algorithm, configuration])
    strategy_name_hash = str(
        int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16)
    )
    outputted_cells = {}
    if algorithm == "OD":
        dataset_path = os.path.join(
            tempfile.gettempdir(), d.name + "-" + strategy_name_hash + ".csv"
        )
        d.write_csv_dataset(dataset_path, d.dataframe)
        params = (
            ["-F", ",", "--statistical", "0.5"]
            + ["--" + configuration[0]]
            + configuration[1:]
            + [dataset_path]
        )
        raha.tools.dBoost.dboost.imported_dboost.run(params)
        algorithm_results_path = dataset_path + "-dboost_output.csv"
        if os.path.exists(algorithm_results_path):
            ocdf = pd.read_csv(
                algorithm_results_path,
                sep=",",
                header=None,
                encoding="utf-8",
                dtype=str,
                keep_default_na=False,
                low_memory=False,
            ).apply(lambda x: x.str.strip())
            for i, j in ocdf.values.tolist():
                if int(i) > 0:
                    outputted_cells[(int(i) - 1, int(j))] = ""
            os.remove(algorithm_results_path)
        os.remove(dataset_path)
    elif algorithm == "PVD":
        ch = configuration
        for attribute in d.dataframe.columns:
            j = d.dataframe.columns.get_loc(attribute)
            for i, value in d.dataframe[attribute].items():
                try:
                    if len(re.findall("[" + ch + "]", value, re.UNICODE)) > 0:
                        outputted_cells[(i, j)] = ""
                except Exception:
                    continue
    elif algorithm == "RVD":
        d_col_list = d.dataframe.columns.tolist()
        configuration_list = []
        for col_idx in range(len(d_col_list)):
            if configuration == "FCD":
                # 1st to col
                configuration_list.append((d_col_list[0], d_col_list[col_idx]))

            elif configuration == "RND":
                # direct neighbours to col
                if col_idx != 0:
                    configuration_list.append(
                        (d_col_list[col_idx - 1], d_col_list[col_idx])
                    )
                else:
                    configuration_list.append(
                        (d_col_list[col_idx], d_col_list[col_idx])
                    )
            elif configuration == "LND":
                if col_idx != len(d_col_list) - 1:
                    configuration_list.append(
                        (d_col_list[col_idx], d_col_list[col_idx + 1])
                    )
                else:
                    configuration_list.append(
                        (d_col_list[col_idx], d_col_list[col_idx])
                    )
        for conf in configuration_list:
            l_attribute, r_attribute = conf
            l_j = d.dataframe.columns.get_loc(l_attribute)
            r_j = d.dataframe.columns.get_loc(r_attribute)
            value_dictionary = {}
            for i, row in d.dataframe.iterrows():
                if row[l_attribute]:
                    if row[l_attribute] not in value_dictionary:
                        value_dictionary[row[l_attribute]] = {}
                    if row[r_attribute]:
                        value_dictionary[row[l_attribute]][row[r_attribute]] = 1
            for i, row in d.dataframe.iterrows():
                if (
                    row[l_attribute] in value_dictionary
                    and len(value_dictionary[row[l_attribute]]) > 1
                ):
                    outputted_cells[(i, l_j)] = ""
                    outputted_cells[(i, r_j)] = ""
    elif algorithm == "KBVD":
        outputted_cells = raha.tools.KATARA.katara.run(d, configuration)
    detected_cells_list = list(outputted_cells.keys())
    strategy_profile = {
        "name": strategy_name,
        "output": detected_cells_list,
        "runtime": time.time() - start_time,
    }
    if self.SAVE_RESULTS:
        pickle.dump(
            strategy_profile,
            open(
                os.path.join(
                    d.results_folder,
                    "strategy-profiling",
                    strategy_name_hash + ".dictionary",
                ),
                "wb",
            ),
        )
    if self.VERBOSE:
        print(
            "{} cells are detected by {}.".format(
                len(detected_cells_list), strategy_name
            )
        )
    return strategy_profile
