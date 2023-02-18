import os
import time
from configparser import ConfigParser

from pyspark.sql import SparkSession

from check_results import evaluate_pyspark
from column_clustering import column_clustering_pyspark
from dataset_clustering import cluster_datasets_pyspark
from error_detector import error_detector_pyspark
from extract_labels import generate_labels_pyspark
from handle_data import generate_csv_paths
from raha_features import generate_raha_features_pyspark


def run_experiments(
    exp_name: str,
    exp_num: int,
    labeling_budget: int,
    config: ConfigParser,
) -> None:
    """_summary_

    Args:
        exp_name (str): _description_
        exp_num (int): _description_
        labeling_budget (int): _description_
        config (ConfigParser): _description_
    """
    logger.warn("Creating experiment output directory")
    experiment_output_path = os.path.join(config["DIRECTORIES"]["output_dir"], exp_name)
    experiment_output_path = os.path.join(experiment_output_path, str(exp_num))
    if not os.path.exists(experiment_output_path):
        os.makedirs(experiment_output_path)

    logger.warn("Genereting CSV paths")
    module_start_time = time.time()
    csv_paths_df = generate_csv_paths(config["DIRECTORIES"]["sandbox_dir"]).cache()
    module_end_time = time.time()
    logger.warn(
        f"Execution time: Genereting CSV paths: {module_end_time - module_start_time:.2f}s"
    )

    logger.warn("Generating labels")
    module_start_time = time.time()
    label_path = os.path.join(
        experiment_output_path, config["DIRECTORIES"]["label_output_filename"]
    )
    generate_labels_pyspark(
        csv_paths_df=csv_paths_df,
        labels_path=label_path,
        extract_labels_enabled=int(config["EXPERIMENTS"]["extract_labels_enabled"]),
    )
    module_end_time = time.time()
    logger.warn(
        f"Execution time: Generating labels: {module_end_time - module_start_time:.2f}s"
    )

    logger.warn("Grouping tables")
    module_start_time = time.time()
    table_grouping_path = os.path.join(
        experiment_output_path,
        config["DIRECTORIES"]["table_clustering_output_filename"],
    )
    cluster_datasets_pyspark(
        csv_paths_df=csv_paths_df,
        output_path=table_grouping_path,
        table_grouping_enabled=int(config["EXPERIMENTS"]["table_clustering_enabled"]),
        auto_clustering_enabled=int(
            config["TABLE_GROUPING"]["auto_clustering_enabled"]
        ),
    )
    module_end_time = time.time()
    logger.warn(
        f"Execution time: Grouping tables: {module_end_time - module_start_time:.2f}s"
    )

    logger.warn("Grouping columns")
    module_start_time = time.time()
    column_grouping_path = os.path.join(
        experiment_output_path,
        config["DIRECTORIES"]["column_clustering_output_filename"],
    )
    grouped_column_groups_path = os.path.join(
        experiment_output_path,
        config["DIRECTORIES"]["grouped_column_groups_output_filename"],
    )
    column_clustering_pyspark(
        csv_paths_df=csv_paths_df,
        table_cluster_df=spark.read.parquet(table_grouping_path),
        column_groups_path=column_grouping_path,
        grouped_column_groups_path=grouped_column_groups_path,
        column_grouping_enabled=int(config["EXPERIMENTS"]["column_clustering_enabled"]),
        auto_clustering_enabled=int(
            config["COLUMN_GROUPING"]["auto_clustering_enabled"]
        ),
        labeling_budget=labeling_budget,
        seed=int(config["EXPERIMENTS"]["seed"]),
    )
    module_end_time = time.time()
    logger.warn(
        f"Execution time: Grouping columns: {module_end_time - module_start_time:.2f}s"
    )

    logger.warn("Creating Raha features")
    module_start_time = time.time()
    raha_features_path = os.path.join(
        experiment_output_path,
        config["DIRECTORIES"]["cell_features_output_filename"],
    )
    generate_raha_features_pyspark(
        csv_paths_df=csv_paths_df,
        grouped_column_groups_df=spark.read.parquet(grouped_column_groups_path),
        column_grouping_df=spark.read.parquet(column_grouping_path),
        raha_features_path=raha_features_path,
        cell_feature_generator_enabled=int(
            config["EXPERIMENTS"]["cell_feature_generator_enabled"]
        ),
    )
    module_end_time = time.time()
    logger.warn(
        f"Execution time: Creating Raha features: {module_end_time - module_start_time:.2f}s"
    )
    csv_paths_df.unpersist()

    logger.warn("Detecting Errors")
    module_start_time = time.time()
    error_prediction_path = os.path.join(
        os.path.join(
            experiment_output_path,
            f"results_exp_{exp_num}_labels_{labeling_budget}",
        ),
        "error_predictions.parquet",
    )
    error_detector_pyspark(
        result_path=error_prediction_path,
        labeling_budget=labeling_budget,
        raha_features_df=spark.read.parquet(raha_features_path),
        labels_df=spark.read.parquet(label_path),
        seed=int(config["EXPERIMENTS"]["seed"]),
    )
    module_end_time = time.time()
    logger.warn(
        f"Execution time: Detecting Errors: {module_end_time - module_start_time:.2f}s"
    )

    logger.warn("Evaluating")
    evaluate_pyspark(
        spark.read.parquet(error_prediction_path),
        spark.read.parquet(label_path),
        result_path=os.path.join(
            experiment_output_path,
            f"results_exp_{exp_num}_labels_{labeling_budget}",
        ),
    )


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    # Workaround for logging. At log level INFO our log messages will be lost.
    # Python's logging module does not work with pyspark.
    spark.sparkContext.setLogLevel("WARN")
    log4j_logger = spark._jvm.org.apache.log4j
    logger = log4j_logger.LogManager.getLogger(__name__)
    logger.warn("Pyspark initialized")

    logger.warn("Reading config")
    configs = ConfigParser()
    configs.read("Configs/config.ini")

    logger.warn("Creating output directory")
    if not os.path.exists(configs["DIRECTORIES"]["output_dir"]):
        os.makedirs(configs["DIRECTORIES"]["output_dir"])

    logger.warn("Starting experiments")
    # TODO: outsoure running the different experiments in python. Maybe better solution to submit each experiment individually for cluster execution.
    for exp_number in [
        int(number)
        for number in configs["EXPERIMENTS"]["experiment_numbers"].split(",")
    ]:
        for number_of_labels in [
            int(number)
            for number in configs["EXPERIMENTS"]["number_of_labels_list"].split(",")
        ]:
            logger.warn(
                f"Runing experiment: Number:{exp_number} Labels:{number_of_labels}"
            )
            experiment_start_time = time.time()
            run_experiments(
                exp_name=configs["EXPERIMENTS"]["exp_name"],
                labeling_budget=number_of_labels,
                exp_num=exp_number,
                config=configs,
            )
            experiment_end_time = time.time()
            logger.warn(
                f"Execution time experiment: Number:{exp_number} Labels:{number_of_labels}, Time:{experiment_end_time - experiment_start_time:.2f}s"
            )
    spark.stop()
    exit()
