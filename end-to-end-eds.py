import os
import dataset_clustering
import extract_ground_truth
import cols_grouping
import logging
import ed_twolevel_rahas_features

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

input_sandbox = "/Users/fatemehahmadi/Documents/Github-Private/Fatemeh/MVP/raha-datasets"
output_files_path = "outputs"
sandbox_name = "raha-datasets"
output_path = os.path.join(output_files_path,  sandbox_name)
context_df_path = os.path.join(output_path, "context_df.csv")

try:
    os.mkdir(output_path)
except OSError as error:
    logger.warning(error) 

# # Clustering datasets
# logger.info("Clustering datasets started")
# dataset_clustering.cluster_datasets(input_sandbox, context_df_path)

# # Extracting Ground Truth
# logger.info("Extracting ground truth started")
# extract_ground_truth.extract_gt(input_sandbox, os.path.join(output_path, "gt.pickle"))

# Column Folding
logger.info("Column folding started")
cols_grouping.col_folding(context_df_path, input_sandbox, output_path, 1)

# # EDS
# logger.info("error detection started")
# ed_twolevel_rahas_features.main



