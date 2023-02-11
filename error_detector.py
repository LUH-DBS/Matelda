import math
import random

from typing import Dict, Tuple
from functools import reduce
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import GBTClassifier

import pyspark.sql.functions as F


def error_detector_pyspark(
    labeling_budget: int,
    result_path: str,
    raha_features_df: DataFrame,
    labels_df: DataFrame,
    column_grouping_df: DataFrame,
    table_grouping_df: DataFrame,
    seed: int,
) -> DataFrame:
    """_summary_

    Args:
        labeling_budget (int): _description_
        result_path (str): _description_
        raha_features_df (DataFrame): _description_
        labels_df (DataFrame): _description_
        column_grouping_df (DataFrame): _description_
        table_grouping_df (DataFrame): _description_
        seed (int): _description_

    Returns:
        DataFrame: _description_
    """
    spark = SparkSession.getActiveSession()
    log4j_logger = spark._jvm.org.apache.log4j
    logger = log4j_logger.LogManager.getLogger(__name__)

    prediction_df = predict_errors(
        column_grouping_df=column_grouping_df,
        table_grouping_df=table_grouping_df,
        raha_features_df=raha_features_df,
        labels_df=labels_df,
        n_labels=labeling_budget,
        seed=seed,
        logger=logger,
        spark=spark,
    )

    logger.warn("Writing error dectection result to disk.")
    prediction_df.select("table_id", "column_id", "row_id", "prediction").write.parquet(
        result_path, mode="overwrite"
    )
    return prediction_df


def predict_errors(
    column_grouping_df: DataFrame,
    table_grouping_df: DataFrame,
    raha_features_df: DataFrame,
    labels_df: DataFrame,
    n_labels: int,
    seed: int,
    logger,
) -> DataFrame:
    """_summary_

    Args:
        column_grouping_df (DataFrame): _description_
        table_grouping_df (DataFrame): _description_
        results_path (str): _description_
        raha_features_df (DataFrame): _description_
        labels_df (DataFrame): _description_
        n_labels (int): _description_
        seed (int): _description_
        logger (_type_): _description_

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, DataFrame]: _description_
    """
    predictions = []

    logger.warn("Joining column cluster, raha features, table grouping and labels")
    x_all_df = (
        raha_features_df.join(column_grouping_df, ["table_id", "column_id"], "inner")
        .join(table_grouping_df, ["table_id"], "inner")
        .join(labels_df, ["table_id", "column_id", "row_id"])
    )

    cluster_combinations = (
        x_all_df.select("table_cluster", "col_cluster")
        .distinct()
        .rdd.map(lambda x: (x.table_cluster, x.col_cluster))
        .collect()
    )

    logger.warn("Splitting labeling budget between column clusters")
    n_cell_clusters_per_col_cluster_dict = split_labeling_budget(
        n_labels, cluster_combinations
    )

    # TODO: is here an way to espress this in pyspark?
    for t_idx, c_idx in sorted(cluster_combinations):
        logger.warn(
            "Table cluster {}, column cluster {}: Start processing".format(t_idx, c_idx)
        )
        cluster_df = x_all_df.where(
            (x_all_df.col_cluster == c_idx) & (x_all_df.table_cluster == t_idx)
        )

        logger.warn(
            "Table cluster {}, column cluster {}: Cells {}".format(
                t_idx, c_idx, cluster_df.count()
            )
        )

        logger.warn(
            "Table cluster {}, column cluster {}: Sampling labels for expected {} label clusters".format(
                t_idx, c_idx, n_cell_clusters_per_col_cluster_dict[(t_idx, c_idx)]
            )
        )

        (cluster_samples_df, sampling_prediction_df,) = sampling_labeling(
            cluster_df,
            n_cell_clusters_per_col_cluster_dict[(t_idx, c_idx)],
            seed,
        )

        logger.warn(
            "Table cluster {}, column cluster {}: Created {} label cluster".format(
                t_idx,
                c_idx,
                cluster_samples_df.count(),
            )
        )

        logger.warn(
            "Table cluster {}, column cluster {}: Label propagation".format(
                t_idx, c_idx
            )
        )

        cluster_df = label_propagation(
            cluster_df, cluster_samples_df, sampling_prediction_df
        )

        logger.warn(
            "Table cluster {}, column cluster {}: Training detection classfier".format(
                t_idx, c_idx
            )
        )

        gb_classifier = GBTClassifier(
            featuresCol="features",
            labelCol="ground_truth_propagated",
            seed=seed,
        )
        gb_classifier_model = gb_classifier.fit(
            cluster_df,
        )

        logger.warn(
            "Table cluster {}, column cluster {}: Predicting errors".format(
                t_idx, c_idx
            )
        )
        predictions.append(gb_classifier_model.transform(cluster_df))

    return reduce(DataFrame.unionAll, predictions)


def split_labeling_budget(
    labeling_budget: int, cluster_combinations: Tuple[int, int]
) -> Dict[int, int]:
    """_summary_

    Args:
        labeling_budget (int): _description_
        cluster_combinations (Tuple[int, int]): _description_

    Returns:
        Dict[int, int]: _description_
    """
    # TODO: what is happening if our labeling budget is smaller than our column cluster?
    n_cell_clusters_per_col_cluster = math.floor(
        labeling_budget / len(cluster_combinations)
    )
    n_cell_clusters_per_col_cluster_dict = {
        col_cluster: n_cell_clusters_per_col_cluster
        for col_cluster in cluster_combinations
    }

    while sum(n_cell_clusters_per_col_cluster_dict.values()) < labeling_budget:
        rand = random.randint(0, len(cluster_combinations) - 1)
        n_cell_clusters_per_col_cluster_dict[cluster_combinations[rand]] += 1

    return n_cell_clusters_per_col_cluster_dict


def label_propagation(
    cluster_df: DataFrame,
    cluster_samples_df: DataFrame,
    predictions_df: DataFrame,
) -> DataFrame:
    """_summary_

    Args:
        cluster_df (DataFrame): _description_
        cluster_samples_df (DataFrame): _description_
        predictions_df (DataFrame): _description_

    Returns:
        DataFrame: _description_
    """
    propagated_df = (
        cluster_df.join(
            predictions_df.select("table_id", "column_id", "row_id", "prediction"),
            ["table_id", "column_id", "row_id"],
        ).join(
            cluster_samples_df.select(
                "prediction", F.col("ground_truth").alias("ground_truth_propagated")
            ),
            "prediction",
        )
    ).withColumnRenamed("prediction", "label_cluster_prediction")

    return propagated_df


def sampling_labeling(
    x: DataFrame,
    n_cell_clusters_per_col_cluster: int,
    seed: int,
) -> Tuple[DataFrame, DataFrame]:
    """_summary_

    Args:
        x (DataFrame): _description_
        y (DataFrame): _description_
        n_cell_clusters_per_col_cluster (int): _description_
        seed (int): _description_

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: _description_
    """
    bkm = KMeans(k=n_cell_clusters_per_col_cluster, featuresCol="features", seed=seed)
    model = bkm.fit(
        x,
    )

    predictions = model.transform(x).drop("features")

    # Draw random sample from each cluster
    window = Window.partitionBy(predictions["prediction"]).orderBy(F.rand())
    samples_df = (
        predictions.select("*", F.rank().over(window).alias("rank"))
        .filter(F.col("rank") <= 1)
        .drop("rank")
    )

    return samples_df, predictions
