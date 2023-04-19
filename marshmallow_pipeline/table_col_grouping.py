from col_grouping_module.col_grouping import group_cols
from cluster_tables import table_grouping

graph_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/output/results/graph.gpickle"
lake_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/data-gov-sandbox-aggregated"
separated_lake_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/data-gov-sandbox"
table_grouping_dict = table_grouping(graph_path)
col_groups = group_cols(lake_path, table_grouping_dict, separated_lake_path, 2015)