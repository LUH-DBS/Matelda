import math
import random
import os

from typing import Dict, Tuple
from functools import reduce
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.feature import Imputer
from xgboost.spark import SparkXGBClassifier

import pyspark.sql.functions as F


def error_detector_pyspark(
    labeling_budget: int,
    cell_clustering_alg: str,
    result_path: str,
    raha_features_df: DataFrame,
    labels_df: DataFrame,
    column_grouping_df: DataFrame,
    number_of_column_clusters: int,
    seed: int,
    save_intermediate_results: bool,
) -> DataFrame:
    """_summary_

    Args:
        labeling_budget (int): _description_
        cell_clustering_alg (str): _description_
        result_path (str): _description_
        raha_features_df (DataFrame): _description_
        labels_df (DataFrame): _description_
        column_grouping_df (DataFrame): _description_
        number_of_column_clusters (int): _description_
        seed (int): _description_
        save_intermediate_results (bool): _description_

    Returns:
        DataFrame: _description_
    """
    spark = SparkSession.getActiveSession()
    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)
    prediction_df = predict_errors(
        column_grouping_df=column_grouping_df,
        results_path=result_path,
        raha_features_df=raha_features_df,
        labels_df=labels_df,
        n_labels=labeling_budget,
        number_of_clusters=number_of_column_clusters,
        cell_clustering_alg=cell_clustering_alg,
        seed=seed,
        logger=logger,
        spark=spark,
    )

    logger.warn("Writing error dectection result to disk.")
    prediction_df = prediction_df.drop("probability", "rawPrediction")
    if save_intermediate_results:
        prediction_df.write.parquet(
            os.path.join(result_path, "error_predictions.parquet"), mode="overwrite"
        )
    return prediction_df


def predict_errors(
    column_grouping_df: DataFrame,
    results_path: str,
    raha_features_df: DataFrame,
    labels_df: DataFrame,
    n_labels: int,
    number_of_clusters: int,
    cell_clustering_alg: str,
    seed: int,
    logger,
    spark,
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """_summary_

    Args:
        column_grouping_df (DataFrame): _description_
        results_path (str): _description_
        raha_features_df (DataFrame): _description_
        labels_df (DataFrame): _description_
        n_labels (int): _description_
        number_of_clusters (int): _description_
        cell_clustering_alg (str): _description_
        seed (int): _description_
        logger (_type_): _description_
        spark (_type_): _description_

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, DataFrame]: _description_
    """
    predictions = []

    logger.warn("Splitting labeling budget between column clusters")
    n_cell_clusters_per_col_cluster_dict = split_labeling_budget(
        n_labels, number_of_clusters
    )

    logger.warn("Joining column cluster with raha features")
    x_test_df = raha_features_df.join(
        column_grouping_df, ["table_id", "column_id"], "inner"
    )
    y_test_df = labels_df.join(column_grouping_df, ["table_id", "column_id"], "inner")

    # TODO: is here an way to espress this in pyspark?
    for c_idx in range(number_of_clusters):
        logger.warn("Processing column cluster {}".format(c_idx))
        cluster_df = x_test_df.where(x_test_df.col_cluster == c_idx)
        cluster_label_df = y_test_df.where(y_test_df.col_cluster == c_idx)
        if len(cluster_df.head(1)) == 0:
            logger.warn("Column cluster {} is empty".format(c_idx))
            continue

        (
            cluster_samples_df,
            cluster_samples_labels_df,
            predictions_df,
        ) = sampling_labeling(
            cluster_df,
            cluster_label_df,
            n_cell_clusters_per_col_cluster_dict[c_idx],
            cell_clustering_alg,
            seed,
            logger,
        )
        cluster_samples_labels_df.show()
        # Save temp results
        # cluster_samples_df.write.parquet(os.path.join(results_path, str(c_idx) + "_samples.parquet"), mode="overwrite")
        # cluster_samples_labels_df.write.parquet(os.path.join(results_path, str(c_idx) +"_samples_labels.parquet"), mode="overwrite")
        # predictions_df.write.parquet(os.path.join(results_path, str(c_idx) +"_clustering.parquet"), mode="overwrite")

        y_train = label_propagation(
            cluster_label_df,
            cluster_samples_labels_df,
            predictions_df,
            logger,
        ).drop("prediction", "col_cluster")

        # TODO: do we need an imputer? If there are missing values the sampling label method would already crashed
        # imputer = Imputer(strategy='mode')
        # imputer.setInputCol("features")
        # imputer_model = imputer.fit(x_train)
        # x_train = imputer_model.transform(x_train)
        # x_train.show()
        # xgboost for spark is experimental feature
        logger.warn("Training detection classfier for column cluster {}".format(c_idx))
        xgb_classifier = SparkXGBClassifier(
            features_col="features",
            label_col="ground_truth",
            num_workers=spark.sparkContext.defaultParallelism,
            verbose=0,
            random_state=seed,
        )
        xgb_classifier_model = xgb_classifier.fit(
            cluster_df.join(y_train, ["table_id", "column_id", "row_id"], "inner")
        )
        logger.warn("Predicting errors")
        predictions.append(xgb_classifier_model.transform(cluster_df))
        # logger.warn("Saving classifier")
        # xgb_classifier_model.save(
        #    os.path.join(result_path, "xgboost-classifier-pyspark-model" + str(c_idx)),
        # )
        # Does not overwrite old savings

    return (reduce(DataFrame.unionAll, predictions),)


def split_labeling_budget(
    labeling_budget: int, number_of_clusters: int
) -> Dict[int, int]:
    """_summary_

    Args:
        labeling_budget (int): _description_
        number_of_clusters (int): _description_

    Returns:
        Dict[int, int]: _description_
    """
    n_cell_clusters_per_col_cluster = math.floor(labeling_budget / number_of_clusters)
    n_cell_clusters_per_col_cluster_dict = {
        col_cluster: n_cell_clusters_per_col_cluster
        for col_cluster in range(number_of_clusters)
    }

    while sum(n_cell_clusters_per_col_cluster_dict.values()) < labeling_budget:
        rand = random.randint(0, number_of_clusters - 1)
        n_cell_clusters_per_col_cluster_dict[rand] += 1

    return n_cell_clusters_per_col_cluster_dict


def label_propagation(
    cluster_label_df: DataFrame,
    cluster_samples_labels_df: DataFrame,
    predictions_df: DataFrame,
    logger,
) -> DataFrame:
    """_summary_

    Args:
        cluster_label_df (DataFrame): _description_
        cluster_samples_labels_df (DataFrame): _description_
        predictions_df (DataFrame): _description_
        logger (_type_): _description_

    Returns:
        DataFrame: _description_
    """
    logger.warn("Propagating label")

    y_train = (
        cluster_label_df.drop("ground_truth")
        .join(
            predictions_df.select("table_id", "column_id", "row_id", "prediction"),
            ["table_id", "column_id", "row_id"],
        )
        .join(
            cluster_samples_labels_df.select("prediction", "ground_truth"), "prediction"
        )
    )

    return y_train


def sampling_labeling(
    x: DataFrame,
    y: DataFrame,
    n_cell_clusters_per_col_cluster: int,
    cells_clustering_alg: str,
    seed: int,
    logger,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """_summary_

    Args:
        x (DataFrame): _description_
        y (DataFrame): _description_
        n_cell_clusters_per_col_cluster (int): _description_
        cells_clustering_alg (str): _description_
        seed (int): _description_
        logger (_type_): _description_

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: _description_
    """
    logger.warn("Clustering cluster values")

    if cells_clustering_alg == "km":
        kmeans = KMeans(k=n_cell_clusters_per_col_cluster, initMode="k-means||")
        kmeans.setSeed(seed)
        model = kmeans.fit(x)
    elif (
        cells_clustering_alg == "hac"
    ):  # TODO: Bisecting k-means is a kind of hierarchical clustering
        # TODO: Why slow?
        bkm = BisectingKMeans(k=n_cell_clusters_per_col_cluster)
        bkm.setSeed(seed)
        model = bkm.fit(x)

    predictions = model.transform(x).drop("features")

    logger.warn("Drawing samples")
    # Draw random sample from each cluster
    window = Window.partitionBy(predictions["prediction"]).orderBy(F.rand())
    samples_df = (
        predictions.select("*", F.rank().over(window).alias("rank"))
        .filter(F.col("rank") <= 1)
        .drop("rank")
    )
    samples_df.show()

    labels_df = y.join(
        samples_df.select("table_id", "column_id", "row_id", "prediction"),
        ["table_id", "column_id", "row_id"],
        how="inner",
    )

    return samples_df, labels_df, predictions
