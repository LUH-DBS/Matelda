from pyspark.sql import SparkSession, DataFrame
import os

from configparser import ConfigParser

from extract_labels import generate_labels_pyspark
from dataset_clustering import cluster_datasets_pyspark

def generate_csv_paths(sandbox_path: str) -> DataFrame:
    csv_paths = []
    sandbox_children_path = [(os.path.join(sandbox_path, dir), dir) for dir in os.listdir(sandbox_path)]
    table_id = 0
    for child_path, parent in sandbox_children_path:
        if os.path.isdir(child_path) and not child_path.startswith("."):
            table_dirs = [(os.path.join(child_path, dir), dir) for dir in os.listdir(child_path)]
            for table_path, table in table_dirs:
                if os.path.isdir(table_path) and not table_path.startswith("."):
                    dirty_path =  table_path + "/dirty.csv"
                    clean_path = table_path + "/" + table + ".csv"
                    if os.path.exists(dirty_path) and os.path.exists(clean_path) and os.path.isfile(dirty_path) and os.path.isfile(clean_path):
                        csv_paths.append((table_id, dirty_path, clean_path, table, parent))
                        table_id += 1
    csv_paths_df = spark.createDataFrame(data=csv_paths, schema = ['table_id', 'dirty_path', 'clean_path','table_name', 'parent'])
    del csv_paths
    return csv_paths_df

def run_experiments(sandbox_path, output_path, exp_name, exp_number, extract_labels_enabled, table_grouping_enabled,
                    column_grouping_enabled, labeling_budget, cell_clustering_alg, cell_feature_generator_enabled):
    logger.warn("Genereting CSV paths")
    csv_paths_df = generate_csv_paths(sandbox_path)

    logger.warn("Generating labels")
    labels_df = generate_labels_pyspark(csv_paths_df, os.path.join(output_path, configs["DIRECTORIES"]["labels_filename"]), extract_labels_enabled)
    labels_df.show()

    logger.warn("Creating experiment output directory")
    experiment_output_path = os.path.join(output_path, exp_name)
    if not os.path.exists(experiment_output_path):
        os.makedirs(experiment_output_path)

    logger.warn("Grouping tables")
    table_grouping_df = cluster_datasets_pyspark(csv_paths_df, os.path.join(output_path, configs["DIRECTORIES"]["table_grouping_output_filename"]), table_grouping_enabled, configs["TABLE_GROUPING"]["auto_clustering_enabled"])



if __name__ == "__main__":
    spark = SparkSession.builder.appName("ED-Scale").getOrCreate()
    # Workaround for logging. At log level INFO our log messages will be lost. Python's logging module does not work with pyspark.
    spark.sparkContext.setLogLevel("WARN")
    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)
    logger.warn("Pyspark initialized")

    logger.warn("Reading config")
    configs = ConfigParser()
    configs.read("config.ini")

    logger.warn("Creating output directory")
    if not os.path.exists(configs["DIRECTORIES"]["output_dir"]):
        os.makedirs(configs["DIRECTORIES"]["output_dir"])

    logger.warn("Starting experiments")
    # TODO: outsoure running the different experiments in python. Maybe better solution to submit each experiment individually for cluster execution.
    for exp_number in [int(number) for number in configs['EXPERIMENTS']['experiment_numbers'].split(',')]:
        for number_of_labels in [int(number) for number in configs['EXPERIMENTS']['number_of_labels_list'].split(',')]:
            logger.warn("Runing experiment: Number:{} Labels:{}".format(exp_number, number_of_labels))
            run_experiments(configs["DIRECTORIES"]["sandbox_dir"], configs["DIRECTORIES"]["output_dir"], configs['EXPERIMENTS']['exp_name'], exp_number, int(configs['EXPERIMENTS']['extract_labels_enabled']), int(configs['EXPERIMENTS']['table_grouping_enabled']), int(configs['EXPERIMENTS']['column_grouping_enabled']), number_of_labels, configs["CLUSTERING"]["cells_clustering_alg"], int(configs['EXPERIMENTS']['cell_feature_generator_enabled']))