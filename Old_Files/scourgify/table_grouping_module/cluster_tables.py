import networkx as nx
import networkx.algorithms.community as nx_comm


def table_grouping(graph_path):
    print("table_grouping")
    print("community detection")
    G = nx.read_gpickle(graph_path)
    comp = nx_comm.louvain_communities(G)

    table_group_dict = {}
    table_group_dict_key = 0
    # Print the partitions
    print("creating table_group_dict")
    for community in comp:
        table_group_dict[table_group_dict_key] = []
        for table in community:
            table_group_dict[table_group_dict_key].append(table)
        table_group_dict_key += 1
    return table_group_dict
