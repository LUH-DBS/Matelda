import hashlib
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
import time
from functools import partial

import numpy as np
import pandas as pd
import raha
from Levenshtein import distance
from scipy.spatial.distance import pdist, squareform


def _strategy_runner_process(self, args):
    """
    This method runs an error detection strategy in a parallel process.
    """
    # try:
    logging.info("_strategy_runner_process: Running strategy: %s", args)
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
                low_memory=False,
            ).apply(lambda x: x.str.strip())
            for i, j in ocdf.values.tolist():
                if int(i) > 0:
                    outputted_cells[(int(i) - 1, int(j))] = ""
            os.remove(algorithm_results_path)
        os.remove(dataset_path)
    elif algorithm == "PVD":
        attribute, ch = configuration
        j = d.dataframe.columns.get_loc(attribute)
        for i, value in d.dataframe[attribute].items():
            try:
                if len(re.findall("[" + ch + "]", value, re.UNICODE)) > 0:
                    outputted_cells[(i, j)] = ""
            except:
                continue
    elif algorithm == "RVD":
        d_col_list = d.dataframe.columns.tolist()
        configuration_list = []
        for col_idx, _ in enumerate(d_col_list):
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
        logging.debug(
            "%s cells are detected by %s", len(detected_cells_list), strategy_name
        )
    # except Exception as e:
    #     logging.error(e)
    #     logging.error("Error in _strategy_runner_process in table: %s, args: %s", d.name, args)
    return strategy_profile


def run_strategies(self, d, char_set, pool):
    """
    This method runs (all or the promising) strategies.
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
                    logging.debug("OD configurations: %s", len(configuration_list))

                elif algorithm_name == "PVD":
                    configuration_list = []
                    for j, attribute in enumerate(d.dataframe.columns):
                        column_group_data = char_set[j]
                        characters_dictionary = {ch: 1 for ch in column_group_data}
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
                    logging.debug("RVD configurations: %s", len(configuration_list))

            random.shuffle(algorithm_and_configurations)
            # strategy_profiles_list = []
            # for [d, algorithm, configuration] in algorithm_and_configurations:
            #     strategy_profiles_list.append(_strategy_runner_process(d, [d, algorithm, configuration]))
            # multiprocessing.freeze_support()
            # logging.debug("len algorithm_and_configurations: %s", len(algorithm_and_configurations))
            # pool = multiprocessing.Pool(64)
            _strategy_runner_process_ = partial(_strategy_runner_process, d)
            strategy_profiles_list = pool.map(
                _strategy_runner_process_, algorithm_and_configurations
            )
            # pool.close()
            # pool.join()
            logging.debug(
                "%%%%%%%%%%%%%%%%%%%%%%All strategies are run on the dataset.%%%%%%%%%%%%%%%%%%%%%%"
            )
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
        logging.debug("%s strategy profiles are collected.", len(d.strategy_profiles))


def generate_features(self, d, char_set_dict):
    """
    This method generates features.
    """
    columns_features_list = []
    for j in range(d.dataframe.shape[1]):
        strategy_profiles = []
        for strategy_profile in d.strategy_profiles:
            strategy = json.loads(strategy_profile["name"])
            if strategy[0] == "PVD":
                if strategy[1][0] == d.dataframe.columns[j]:
                    strategy_profiles.append(strategy_profile)
            else:
                strategy_profiles.append(strategy_profile)
        parsed_keys = [
            json.loads(str(key["name"])) for key in strategy_profiles
        ]  # Parse the keys into Python objects
        sorted_keys = sorted(parsed_keys)
        sorted_strategy_profiles = dict()
        for key in sorted_keys:
            for strategy_profile in strategy_profiles:
                if json.loads(str(strategy_profile["name"])) == key:
                    sorted_strategy_profiles[str(key)] = strategy_profile["output"]
        strategy_profiles = [str(k) for k in sorted_keys]
        feature_vectors = np.zeros((d.dataframe.shape[0], len(strategy_profiles)))
        for strategy_index, strategy_name in enumerate(sorted_strategy_profiles):
            logging.debug(
                "******************************Generating features for strategy: %s",
                strategy_name,
            )
            if eval(strategy_name)[0] in self.ERROR_DETECTION_ALGORITHMS:
                for cell in sorted_strategy_profiles[strategy_name]:
                    if cell[1] == j:
                        feature_vectors[cell[0], strategy_index] = 1.0

        if self.VERBOSE:
            logging.debug(
                "%s Features are generated for column %s", feature_vectors.shape[1], j
            )

        columns_features_list.append(feature_vectors)

    d.column_features = columns_features_list


def generate_raha_features(parent_path, dataset_name, charsets, dirty_file_name, clean_file_name, pool):
    sp_path = (
        parent_path + "/" + dataset_name + "/" + "raha-baran-results-" + dataset_name
    )
    if os.path.exists(sp_path):
        shutil.rmtree(sp_path)

    detect = raha.Detection()
    dataset_dictionary = {
        "name": dataset_name,
        "path": parent_path + "/" + dataset_name + "/{}".format(dirty_file_name),
        "clean_path": parent_path + "/" + dataset_name + "/{}".format(clean_file_name),
    }
    detect.VERBOSE = False
    d = detect.initialize_dataset(dataset_dictionary)
    d.SAVE_RESULTS = False
    d.VERBOSE = False
    d.ERROR_DETECTION_ALGORITHMS = ["PVD", "OD", "RVD"]
    logging.debug("Dataset is initialized.")
    logging.debug("Dataset name: %s", d.name)
    t1 = time.time()
    # try:
    run_strategies(detect, d, charsets, pool)
    # except Exception as e:
    #     logging.error(e)
    #     logging.error("Error in run_strategies in table: %s", dataset_name)
    t2 = time.time()
    logging.debug("Strategies are run, time: %s", str(t2 - t1))
    t1 = time.time()
    generate_features(detect, d, charsets)
    t2 = time.time()
    logging.debug("Features are generated.")
    logging.debug("Time - generate features: %s", str(t2 - t1))
    return d.column_features
