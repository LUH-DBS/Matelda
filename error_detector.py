import math
import random

from typing import Dict, Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.clustering import KMeans, BisectingKMeans


def error_detector_pyspark(
    labeling_budget: int,
    cell_clustering_alg: str,
    result_path: str,
    raha_features_df: DataFrame,
    labels_df: DataFrame,
    column_grouping_df: DataFrame,
    number_of_column_clusters: int,
):
    spark = SparkSession.getActiveSession()
    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)
    (
        X_train,
        y_train,
        X_test,
        y_test,
        original_data_values,
        n_samples,
    ) = get_train_test_sets(
        column_grouping_df=column_grouping_df,
        results_path=result_path,
        raha_features_df=raha_features_df,
        labels_df=labels_df,
        n_labels=labeling_budget,
        number_of_clusters=number_of_column_clusters,
        cell_clustering_alg=cell_clustering_alg,
        logger=logger,
    )

    # TODO: classify
    # gbt = GBTClassifier()
    # model = gbt.fit()


def get_train_test_sets(
    column_grouping_df: DataFrame,
    results_path,
    raha_features_df: DataFrame,
    labels_df: DataFrame,
    n_labels: int,
    number_of_clusters: int,
    cell_clustering_alg: str,
    logger,
) -> Tuple[list, list, list, list, list, int]:
    """_summary_

    Args:
        column_grouping_df (DataFrame): _description_
        results_path (_type_): _description_
        raha_features_df (DataFrame): _description_
        labels_df (DataFrame): _description_
        n_labels (int): _description_
        number_of_clusters (int): _description_
        cell_clustering_alg (str): _description_
        logger (_type_): _description_

    Returns:
        Tuple[list, list, list, list, list, int]: _description_
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    original_data_values = []
    labels = []

    logger.warn("Splitting labeling budget between column clusters")
    n_cell_clusters_per_col_cluster_dict = split_labeling_budget(
        n_labels, number_of_clusters
    )

    logger.warn("Joining test dataframe")
    x_test_df = raha_features_df.join(
        column_grouping_df, ["table_id", "column_id"], "inner"
    )
    y_test_df = labels_df.join(column_grouping_df, ["table_id", "column_id"], "inner")

    # TODO: is here an way to espress this in pyspark?
    for c_idx in range(number_of_clusters + 1):
        logger.warn("Processing cluster {}".format(c_idx))
        cluster_df = x_test_df.where(x_test_df.col_cluster == c_idx)
        cluster_label_df = y_test_df.where(x_test_df.col_cluster == c_idx)

        if len(cluster_df.head(1)) == 0:
            logger.warn("Cluster {} is empty".format(c_idx))
            continue

        cluster_samples_df, cluster_samples_labels_df = sampling_labeling(
            cluster_df,
            cluster_label_df,
            n_cell_clusters_per_col_cluster_dict[c_idx],
            cell_clustering_alg,
            logger,
        )
        print(cluster_samples_df.show())
        print(cluster_samples_labels_df.show())

        # X_train, y_train = label_propagation(
        #     X_train,
        #     X_tmp,
        #     y_train,
        #     cells_per_cluster,
        #     labels_per_cluster,
        #     logger,
        # )

    return X_train, y_train, X_test, y_test, original_data_values, len(labels)


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
        for col_cluster in range(number_of_clusters + 1)
    }

    while sum(n_cell_clusters_per_col_cluster_dict.values()) < labeling_budget:
        rand = random.randint(0, number_of_clusters - 1)
        n_cell_clusters_per_col_cluster_dict[rand] += 1

    return n_cell_clusters_per_col_cluster_dict


def label_propagation(
    x_train, x_tmp, y_train, cells_per_cluster, labels_per_cluster, logger
):
    logger.warn("Propagating label")
    for key in list(cells_per_cluster.keys()):
        for cell in cells_per_cluster[key]:
            x_train.append(x_tmp[cell])
            y_train.append(labels_per_cluster[key])
    logger.info("Length of X_train: {}".format(len(x_train)))
    return x_train, y_train


def sampling_labeling(
    x: DataFrame,
    y: DataFrame,
    n_cell_clusters_per_col_cluster: int,
    cells_clustering_alg: str,
    logger,
) -> Tuple[DataFrame, DataFrame]:
    """_summary_

    Args:
        x (DataFrame): _description_
        y (DataFrame): _description_
        n_cell_clusters_per_col_cluster (int): _description_
        cells_clustering_alg (str): _description_
        logger (_type_): _description_

    Returns:
        Tuple[DataFrame, DataFrame]: _description_
    """
    logger.warn("Clustering cluster values")

    if cells_clustering_alg == "km":
        kmeans = KMeans(k=n_cell_clusters_per_col_cluster)
        kmeans.setSeed(0)  # TODO: can be a bad seed
        model = kmeans.fit(x)
    elif (
        cells_clustering_alg == "hac"
    ):  # TODO: Bisecting k-means is a kind of hierarchical clustering
        # TODO: Why slow?
        bkm = BisectingKMeans(k=n_cell_clusters_per_col_cluster)
        bkm.setSeed(0)  # TODO: can be a bad seed
        model = bkm.fit(x)

    predictions = model.transform(x).drop("features")

    logger.warn("Drawing samples")

    # TODO: can be expensive
    samples_df = predictions.groupBy("prediction").applyInPandas(
        get_random_sample, predictions.schema
    )

    labels_df = y.join(samples_df, ["table_id", "column_id", "row_id"], how="left")

    return samples_df, labels_df


def get_random_sample(x: DataFrame) -> DataFrame:
    sample = x.rdd.takeSample(withReplacement=False, num=1)
    return (
        x.where(x.table_id == sample["table_id"])
        .where(x.column_id == sample["column_id"])
        .where(x.row_id == sample["row_id"])
    )
