import pandas as pd
import networkx as nx
import sys


def get_comm_score(G, edge):
    if G.nodes[edge[0]]["community"] == G.nodes[edge[1]]["community"]:
        return 1
    else:
        return 0


def get_coeff(edge, preds):
    for pred in preds:
        if (pred[0], pred[1]) == edge or (pred[1], pred[0]) == edge:
            return pred[2]

def apply_prediction(func, ebunch):
    return ((u, v, func(u, v)) for u, v in ebunch)


def SLP_prediction(G, ebunch, clust_coef="ClustCoef", centrality="DegCent"):
    '''
    Link prediction using clustering_coefficient and a centrality measure
    '''
    def predict(u, v):
        # NodeId, ClustCoefDiff, ClustCoefAvg, DegCentDiff, DegCentAVG, EigenCentDiff, EigenCentAVG, ClosenessCentDiff, ClosenessCentAVG, BetweenCentDiff, BetweenCentAVG, PageRankDiff, PageRankAVG

        CCu = float(G.nodes[u][clust_coef])
        CCv = float(G.nodes[v][clust_coef])
        centu = float(G.nodes[u][centrality])
        centv = float(G.nodes[v][centrality])
        #cnbors = list(nx.common_neighbors(G, u, v))
        #neighbors = (sum(_community(G, w, community) == Cu for w in cnbors)
        #             if Cu == Cv else 0)
        #return ((len(cnbors) + (float(centu) + float(centv))) +  neighbors) / (float(CCu) + float(CCv) + 0.01)
        # return (float(centu) + float(centv))*(0.01/2) / (float(CCu) + float(CCv) + 0.01)
        return (float(centu) + float(centv))*(0.01/2) / (float(CCu) + float(CCv) + 0.01)
    return apply_prediction(predict, ebunch)


def SLPC_prediction(G, ebunch, clust_coef="ClustCoef", centrality="DegCent", community="community"):
    '''
    Link prediction using community information, clustering_coefficient and a centrality measure
    '''
    def predict(u, v):
        # NodeId, ClustCoefDiff, ClustCoefAvg, DegCentDiff, DegCentAVG, EigenCentDiff, EigenCentAVG, ClosenessCentDiff, ClosenessCentAVG, BetweenCentDiff, BetweenCentAVG, PageRankDiff, PageRankAVG

        CCu = float(G.nodes[u][clust_coef])
        CCv = float(G.nodes[v][clust_coef])
        centu = float(G.nodes[u][centrality])
        centv = float(G.nodes[v][centrality])
        same_community = 0
        try:
            if G.nodes[u][community] == G.nodes[v][community]:
                same_community = 1
        except Exception as e:
            print(" * ERROR in checking community information for nodes {} and {}\n".format(u, v))
            raise
        return (((float(centu) + float(centv) +  same_community)) * (0.01/3)) / (float(CCu) + float(CCv) + 0.01)
    return apply_prediction(predict, ebunch)



step = int(sys.argv[2])
second_file = int(sys.argv[1])
first_file = second_file - step
if second_file < 10:
    second_file = "0{}".format(second_file)
if first_file < 10:
    first_file = "0{}".format(first_file)


# ## The network edges file path that will be used to extract the given edges features
# input_network_edges_file_path = input("Enter the network edges file path\n   >>> ").replace("\"", "")
# ## Nodes features
# input_nodes_features_file_path = input("Enter nodes features file path\n   >>> ").replace("\"", "")
# ## Nodes community information
# input_nodes_community_info = input("Enter nodes community info file path\n   >>> ").replace("\"", "")
# ## edges that we want to calculate features for them
# input_edges_file_path = input("Enter edges file path\n   >>> ").replace("\"", "")

# ## The network edges file path that will be used to extract the given edges features
# input_network_edges_file_path = "E:/Research/Dataset/IEEE Dataset/IEEE ComputerScience Journals/2019/jour/2000_{}.csv".format(first_file)
# ## Nodes features
# input_nodes_features_file_path = "E:/Research/Dataset/IEEE Dataset/IEEE ComputerScience Journals/2019/jour/2000_{}-output.csv".format(first_file)
# ## Nodes community information
# input_nodes_community_info = "E:/Research/Dataset/IEEE Dataset/IEEE ComputerScience Journals/2019/CommunityInfo/20{}.csv".format(first_file)
# ## edges that we want to calculate features for them
# input_edges_file_path = "E:/Research/Dataset/IEEE Dataset/IEEE ComputerScience Journals/2019/newPredFiles/{}s/diff-{}-{}.txt".format(step, first_file, second_file)
# output_file_path = "E:/Research/Dataset/IEEE Dataset/IEEE ComputerScience Journals/2019/newPredFiles/{}s/features/diff-features-{}-{}.txt".format(step, first_file, second_file)
# # Top-20 IEEE Conferences

## The network edges file path that will be used to extract the given edges features
input_network_edges_file_path = "/home/saman/KW/jour/2000_{}.csv".format(first_file)
## Nodes features
input_nodes_features_file_path = "/home/saman/KW/jour/2000_{}-output.csv".format(first_file)
## Nodes community information
input_nodes_community_info = "jour/CommunityInfo/20{}.csv".format(first_file)
## edges that we want to calculate features for them
input_edges_file_path = "jour/{}s/diff-{}-{}.txt".format(step, first_file, second_file)
output_file_path = "jour/{}s/features/diff-features-{}-{}.txt".format(step, first_file, second_file)

print("Creating network from 2000_{}.csv edges . . .".format(first_file))
network_edges_df = pd.read_csv(input_network_edges_file_path)
network_edges = [(edge[1]["Source"], edge[1]["Target"]) for edge in network_edges_df.iterrows()]
G = nx.Graph()
G.add_edges_from(network_edges)

print("Reading nodes' community info from 20{}.csv . . .".format(first_file))
nodes_comm_info = pd.read_csv(input_nodes_community_info)

print("Reading nodes features from 2000_{}-output.csv . . .".format(first_file))
nodes_features = pd.read_csv(input_nodes_features_file_path)

print("Loading edges for feature calculation from diff-{}-{}.txt . . .".format(step, first_file, second_file))
try:
    edges_df = pd.read_csv(input_edges_file_path)
    output_edges = [(edge[1]["Source"], edge[1]["Target"]) for edge in edges_df.iterrows()]
except Exception as e:
    edges_df = pd.read_csv(input_edges_file_path, names=["Source", "Target"])
    output_edges = [(edge[1]["Source"], edge[1]["Target"]) for edge in edges_df.iterrows()]


## adding Nodes features to the Network (G)
print("Adding nodes features . . .")
nodes_error = []
lnnodes = len(G.nodes)
cntr = 0
for node in G.nodes:
    cntr += 1
    print("\r  {}/{}  | Node:  {}".format(cntr, lnnodes, node), end="")
    try:
        # Eccentricity ClustCoef DegCent EigenCent ClosenessCent BetweenCent HarmonicCent PageRank
        # G.nodes[node]["Degree"] = nodes_features.loc[nodes_features["NodeId"] == node]["Degree"]
        # G.nodes[node]["Eccentricity"] = nodes_features.loc[nodes_features["NodeId"] == node]["Eccentricity"]
        G.nodes[node]["ClustCoef"] = float(nodes_features.loc[nodes_features["NodeId"] == node]["ClustCoef"])
        G.nodes[node]["DegCent"] = float(nodes_features.loc[nodes_features["NodeId"] == node]["DegCent"])
        G.nodes[node]["EigenCent"] = float(nodes_features.loc[nodes_features["NodeId"] == node]["EigenCent"])
        G.nodes[node]["ClosenessCent"] = float(nodes_features.loc[nodes_features["NodeId"] == node]["ClosenessCent"])
        G.nodes[node]["BetweenCent"] = float(nodes_features.loc[nodes_features["NodeId"] == node]["BetweenCent"])
        G.nodes[node]["PageRank"] = float(nodes_features.loc[nodes_features["NodeId"] == node]["PageRank"])
        G.nodes[node]["community"] = int(nodes_comm_info.loc[nodes_comm_info["id"] == node]["modularity_class"])
    except Exception as e:
        nodes_error.append(node)
        pass

print("  |  # Nodes with Errors: {} ".format(len(nodes_error)))

print("Calculating topological features (Jaccard Coefficient) . . .")
jacc_preds = nx.jaccard_coefficient(G, output_edges)
print("Calculating topological features (Resource Allocation Coefficient) . . .")
ra_preds = nx.resource_allocation_index(G, output_edges)
print("Calculating topological features (KNLP coefficient) . . .")
knlp_preds_DegCent = SLP_prediction(G, output_edges, centrality="DegCent")
knlp_preds_EigenCent = SLP_prediction(G, output_edges, centrality="EigenCent")
knlp_preds_ClosenessCent = SLP_prediction(G, output_edges, centrality="ClosenessCent")
knlp_preds_BetweenCent = SLP_prediction(G, output_edges, centrality="BetweenCent")
knlp_preds_PageRank = SLP_prediction(G, output_edges, centrality="PageRank")
print("Calculating topological features (KNLPC coefficient) . . .")
knlpc_preds_DegCent = SLPC_prediction(G, output_edges, centrality="DegCent")
knlpc_preds_EigenCent = SLPC_prediction(G, output_edges, centrality="EigenCent")
knlpc_preds_ClosenessCent = SLPC_prediction(G, output_edges, centrality="ClosenessCent")
knlpc_preds_BetweenCent = SLPC_prediction(G, output_edges, centrality="BetweenCent")
knlpc_preds_PageRank = SLPC_prediction(G, output_edges, centrality="PageRank")


output = pd.DataFrame(columns=["Source", "Target", "S-degree", "T-degree", "S-ClustCoef", "T-ClustCoef",\
"S-DegCent", "S-EigenCent", "T-EigenCent", "S-ClosenessCent", "T-ClosenessCent",\
"S-BetweenCent", "T-BetweenCent", "S-PageRank", "T-PageRank", "JaccCoeff", "RACoeff",\
"KNLP_DegCent", "KNLP_EigenCent", "KNLP_ClosenessCent", "KNLP_BetweenCent", "KNLP_PageRank",\
"KNLPC_DegCent", "KNLPC_EigenCent", "KNLPC_ClosenessCent", "KNLPC_BetweenCent", "KNLPC_PageRank", "community_score"])

print("Adding features to edges . . .")
lnedges = len(output_edges)
for ind, edge in zip(range(lnedges), output_edges):
    print("\r   {}/{}".format(ind+1, lnedges), end="")
    source_node = edge[0]
    target_node = edge[1]
    try:
        output = output.append({
        "Source": source_node, "Target": target_node,\
        "S-degree": int(nodes_features.loc[nodes_features["NodeId"] == source_node]["Degree"]), "T-degree": int(nodes_features.loc[nodes_features["NodeId"] == target_node]["Degree"]),\
        "S-ClustCoef": G.nodes[source_node]["ClustCoef"], "T-ClustCoef": G.nodes[target_node]["ClustCoef"],\
        "S-DegCent": G.nodes[source_node]["DegCent"], "S-DegCent": G.nodes[target_node]["DegCent"],\
        "S-EigenCent": G.nodes[source_node]["EigenCent"], "T-EigenCent": G.nodes[target_node]["EigenCent"],\
        "S-ClosenessCent": G.nodes[source_node]["ClosenessCent"], "T-ClosenessCent": G.nodes[target_node]["ClosenessCent"],\
        "S-BetweenCent": G.nodes[source_node]["BetweenCent"], "T-BetweenCent": G.nodes[target_node]["BetweenCent"],\
        "S-PageRank": G.nodes[source_node]["PageRank"], "T-PageRank": G.nodes[target_node]["PageRank"],\
        "JaccCoeff": get_coeff(edge, jacc_preds), "RACoeff": get_coeff(edge, ra_preds),\
        "KNLP_DegCent": get_coeff(edge, knlp_preds_DegCent), "KNLPC_DegCent": get_coeff(edge, knlpc_preds_DegCent),\
        "KNLP_EigenCent": get_coeff(edge, knlp_preds_EigenCent), "KNLPC_EigenCent": get_coeff(edge, knlpc_preds_EigenCent),\
        "KNLP_ClosenessCent": get_coeff(edge, knlp_preds_ClosenessCent), "KNLPC_ClosenessCent": get_coeff(edge, knlpc_preds_ClosenessCent),\
        "KNLP_BetweenCent": get_coeff(edge, knlp_preds_BetweenCent), "KNLPC_BetweenCent": get_coeff(edge, knlpc_preds_BetweenCent),\
        "KNLP_PageRank": get_coeff(edge, knlp_preds_PageRank), "KNLPC_PageRank": get_coeff(edge, knlpc_preds_PageRank),\
        "community_score": get_comm_score(G, edge)}, ignore_index=True)
    except Exception as e:
        raise

print("\nWriting results to file: diff-features-{}-{}.txt . . .".format(step, first_file, second_file))
output.to_csv(output_file_path, index=False)












#
