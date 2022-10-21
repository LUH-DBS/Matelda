from pyspark.sql import DataFrame, SparkSession


def generate_raha_features_pyspark(
    csv_paths_df: DataFrame,
    output_path: str,
    cell_feature_generator_enabled: int,
    cells_clustering_alg: str,
) -> DataFrame:
    spark = SparkSession.getActiveSession()
    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)

    if cell_feature_generator_enabled == 1:
        logger.warn("Creating Raha features")
    else:
        logger.warn("Loading Raha features from disk")