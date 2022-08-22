import os
import pickle

import app_logger

logger = app_logger.get_logger()


def get_all_results(output_path, results_path):
    with open(os.path.join(results_path, "results_df.pickle"), 'rb') as file:
        results_df = pickle.load(file)
        logger.info("results_df loaded.")
        print(results_df.shape)

    with open(os.path.join(results_path, "scores.pickle"), 'rb') as file:
        scores = pickle.load(file)
        logger.info("Scores loaded.")
        print(scores)

    with open(os.path.join(results_path, "sampled_tuples.pkl"), 'rb') as file:
        labels = pickle.load(file)
        logger.info("sampled_tuples loaded.")
        print(labels)

