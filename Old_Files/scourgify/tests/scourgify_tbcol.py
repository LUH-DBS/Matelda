from scourgify.col_grouping_module.col_grouping import group_cols
from scourgify.table_grouping_module.cluster_tables import table_grouping

graph_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/scourgify/graph.gpickle"
lake_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/scourgify/datalake"
server_lake_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/sandbox-parent/data-gov-sandbox"
table_grouping_dict = table_grouping(graph_path)
col_groups = group_cols(lake_path, table_grouping_dict, server_lake_path)