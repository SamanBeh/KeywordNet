import networkx as nx
import pandas as pd
from sklearn.utils import shuffle
import math


algorithms = ["jaccard", "adamic_adar"]

perf_measures = ["sensitivity","specificity","precision","npv","fnr","fpr","fdr","forate","acc","f1score","mcc"]


def write_pred_proba_to_file(file_name, pred_pos, pred_neg):
    df_pos = pd.DataFrame()
    df_neg = pd.DataFrame()

    df_pos["positive"] = pred_pos
    df_neg["negative"] = pred_neg

    df_out = pd.concat([df_pos, df_neg], axis=1, ignore_index=False)
    # df_out = df_out.fillna("")

    df_out.to_csv("d:/topol/{}".format(file_name), index=False, sep=",")


def calculate_performance(perf_measure, TP, TN, FP, FN):
    try:
        if perf_measure == "sensitivity":
            return TP / (TP + FN)
        elif perf_measure == "specificity":
            return TN / (TN + FP)
        elif perf_measure == "precision":
            return TP / (TP + FP)
        elif perf_measure == "npv":
            return TN / (TN + FN)
        elif perf_measure == "fnr":
            return FN / (FN + TP)
        elif perf_measure == "fpr":
            return FP / (FP + TN)
        elif perf_measure == "fdr":
            return FP / (FP + TP)
        elif perf_measure == "forate":
            return FN / (FN + TN)
        elif perf_measure == "acc":
            return (TP + TN) / (TP + TN + FP + FN)
        elif perf_measure == "f1score":
            return (2 * TP) / (2 * TP + FP + FN)
        elif perf_measure == "mcc":
            return ((TP * TN) - (FP * FN)) / math.sqrt(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    except Exception as e:
        return 0


def get_tp_np(positive_predictions_proba, negative_predictions_proba, threshold):
    pos_predictions = []
    for pp in positive_predictions_proba:
        if pp >= threshold:
            pos_predictions.append(1)
        else:
            pos_predictions.append(0)

    neg_predictions = []
    for pp in negative_predictions_proba:
        if pp < threshold:
            neg_predictions.append(1)
        else:
            neg_predictions.append(0)

    # tp = print("\nTP: {}".format((pos_predictions.count(1)*100)/len(pos_predictions)))
    # tn = print("\nTN: {}".format((neg_predictions.count(1)*100)/len(neg_predictions)))
    tp = (pos_predictions.count(1)*100)/len(pos_predictions)
    fn = 100 - tp
    tn = (neg_predictions.count(1)*100)/len(neg_predictions)
    fp = 100 - tn

    return tp, fp, tn, fn


def get_all_perf_measures(positive_predictions_proba, negative_predictions_proba, threshold):
    all_measures = []
    TP, FP, TN, FN = get_tp_np(positive_predictions_proba, negative_predictions_proba, threshold)
    all_measures = [TP, FP, TN, FN]
    for measure in perf_measures:
        all_measures.append(calculate_performance(perf_measure=measure, TP=TP, FP=FP, TN=TN, FN=FN))
    return all_measures


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




##############################################################
##############################################################
##############################################################


positive_edges_to_predict_path = input("Please enter the positive edges file path you want to predict:\n   >>> ")
negative_edges_to_predict_path = input("Please enter the negative edges file path you want to predict:\n   >>> ")
previous_network_edges_path = input("Please enter the edges file path of previous network snapshot:\n   >>> ")
comm_info_previous_edges_path = input("Please enter the community info file path for previous network snapshot:\n   >>> ")
nodes_info_previous_edges_path = input("Please enter the nodes info file path for previous network snapshot:\n   >>> ")
K = int(input("K(number of iterations to test each prediction algorithm):  "))

print("Reading input files . . .")
positive_edges_to_predict = pd.read_csv(positive_edges_to_predict_path)#.iloc[:1000,:]
negative_edges_to_predict = pd.read_csv(negative_edges_to_predict_path)#.iloc[:1000,:]
previous_network_edges = pd.read_csv(previous_network_edges_path)
comm_info_previous_edges = pd.read_csv(comm_info_previous_edges_path)
nodes_info_previous_edges = pd.read_csv(nodes_info_previous_edges_path)

sources = previous_network_edges.Source
targets = previous_network_edges.Target
edges = [(int(src), int(tgt)) for src,tgt in zip(sources, targets)]
G = nx.Graph()
G.add_edges_from(edges)
# G = nx.read_edgelist(previous_network_edges_path)
# Add community info to nodes
print("Adding nodes information [be patient please, it may take a while.]. . .")
for node in list(G.nodes()):
    # NodeId, ClustCoef, DegCent, EigenCent, ClosenessCent, BetweenCent, PageRank
    G.nodes[node]["community"] = float(comm_info_previous_edges[comm_info_previous_edges["id"] == int(node)]["modularity_class"])
    G.nodes[node]["ClustCoef"] = float(nodes_info_previous_edges[nodes_info_previous_edges["NodeId"] == int(node)]["ClustCoef"])
    G.nodes[node]["DegCent"] = float(nodes_info_previous_edges[nodes_info_previous_edges["NodeId"] == int(node)]["DegCent"])
    G.nodes[node]["EigenCent"] = float(nodes_info_previous_edges[nodes_info_previous_edges["NodeId"] == int(node)]["EigenCent"])
    G.nodes[node]["ClosenessCent"] = float(nodes_info_previous_edges[nodes_info_previous_edges["NodeId"] == int(node)]["ClosenessCent"])
    G.nodes[node]["BetweenCent"] = float(nodes_info_previous_edges[nodes_info_previous_edges["NodeId"] == int(node)]["BetweenCent"])
    G.nodes[node]["PageRank"] = float(nodes_info_previous_edges[nodes_info_previous_edges["NodeId"] == int(node)]["PageRank"])

# Positive edges
print("extracting positive edges . . .")
psources = positive_edges_to_predict.Source
ptargets = positive_edges_to_predict.Target
pedges = [(src, tgt) for src,tgt in zip(psources, ptargets)]

all_folds_df = []
for i in range(K):
    print("Step {}/{}".format(i+1, K))
    df = pd.DataFrame(columns=["algorithm", "TP", "FP", "TN", "FN"] + perf_measures)
    # My negative dataset is much larger than the positive and i decided to
    # shuffle it in every iteration then select a portion of samples as large as
    # my positive samples
    print("\nShuffling negative dataset . . .")
    # shuffle(negative_edges_to_predict)
    # negative_edges_to_predict = negative_edges_to_predict.iloc[:len(positive_edges_to_predict),:]

    # Negative edges
    print("\nextracting negative edges . . .")
    nsources = negative_edges_to_predict.Source
    ntargets = negative_edges_to_predict.Target
    nedges = [(src, tgt) for src,tgt in zip(nsources, ntargets)]
    # edges_to_predict = pedges + nedges

    print("\n Performing predictions on Positive data . . .")
    positive_predictions_proba_jcc  = []
    positive_predictions_proba_ra  = []
    positive_predictions_proba_aa  = []
    positive_predictions_proba_pa  = []
    positive_predictions_proba_cnsh  = []
    positive_predictions_proba_rash  = []
    positive_predictions_proba_wic  = []
    positive_predictions_proba_slp_DegCent  = []
    positive_predictions_proba_slp_EigenCent  = []
    positive_predictions_proba_slp_ClosenessCent  = []
    positive_predictions_proba_slp_BetweenCent  = []
    positive_predictions_proba_slp_PageRank  = []
    positive_predictions_proba_slpc_DegCent  = []
    positive_predictions_proba_slpc_EigenCent  = []
    positive_predictions_proba_slpc_ClosenessCent  = []
    positive_predictions_proba_slpc_BetweenCent  = []
    positive_predictions_proba_slpc_PageRank  = []
    lenedg = len(pedges)
    cntr = 0
    for edge in pedges:
        cntr += 1
        print("\r   {}/{}".format(cntr, lenedg), end="")
        positive_predictions_proba_jcc.append(list(nx.jaccard_coefficient(G, [edge]))[0][2])
        positive_predictions_proba_ra.append(list(nx.resource_allocation_index(G, [edge]))[0][2])
        positive_predictions_proba_aa.append(list(nx.adamic_adar_index(G, [edge]))[0][2])
        positive_predictions_proba_pa.append(list(nx.preferential_attachment(G, [edge]))[0][2])
        positive_predictions_proba_cnsh.append(list(nx.cn_soundarajan_hopcroft(G, [edge]))[0][2]) # needs community information
        positive_predictions_proba_rash.append(list(nx.ra_index_soundarajan_hopcroft(G, [edge]))[0][2]) # needs community information
        positive_predictions_proba_wic.append(list(nx.within_inter_cluster(G, [edge]))[0][2]) # needs community information
        positive_predictions_proba_slp_DegCent.append(list(SLP_prediction(G, [edge], centrality="DegCent"))[0][2])
        positive_predictions_proba_slp_EigenCent.append(list(SLP_prediction(G, [edge], centrality="EigenCent"))[0][2])
        positive_predictions_proba_slp_ClosenessCent.append(list(SLP_prediction(G, [edge], centrality="ClosenessCent"))[0][2])
        positive_predictions_proba_slp_BetweenCent.append(list(SLP_prediction(G, [edge], centrality="BetweenCent"))[0][2])
        positive_predictions_proba_slp_PageRank.append(list(SLP_prediction(G, [edge], centrality="PageRank"))[0][2])
        positive_predictions_proba_slpc_DegCent.append(list(SLPC_prediction(G, [edge], centrality="DegCent"))[0][2]) # needs community information
        positive_predictions_proba_slpc_EigenCent.append(list(SLPC_prediction(G, [edge], centrality="EigenCent"))[0][2]) # needs community information
        positive_predictions_proba_slpc_ClosenessCent.append(list(SLPC_prediction(G, [edge], centrality="ClosenessCent"))[0][2]) # needs community information
        positive_predictions_proba_slpc_BetweenCent.append(list(SLPC_prediction(G, [edge], centrality="BetweenCent"))[0][2]) # needs community information
        positive_predictions_proba_slpc_PageRank.append(list(SLPC_prediction(G, [edge], centrality="PageRank"))[0][2]) # needs community information


    print("\n Performing predictions on Negative data . . .")
    negative_predictions_proba_jcc = []
    negative_predictions_proba_ra = []
    negative_predictions_proba_aa = []
    negative_predictions_proba_pa = []
    negative_predictions_proba_cnsh = []
    negative_predictions_proba_rash = []
    negative_predictions_proba_wic = []
    negative_predictions_proba_slp_DegCent = []
    negative_predictions_proba_slp_EigenCent = []
    negative_predictions_proba_slp_ClosenessCent = []
    negative_predictions_proba_slp_BetweenCent = []
    negative_predictions_proba_slp_PageRank = []
    negative_predictions_proba_slpc_DegCent = []
    negative_predictions_proba_slpc_EigenCent = []
    negative_predictions_proba_slpc_ClosenessCent = []
    negative_predictions_proba_slpc_BetweenCent = []
    negative_predictions_proba_slpc_PageRank = []
    lenedg = len(nedges)
    cntr = 0
    for edge in nedges:
        cntr += 1
        print("\r   {}/{}".format(cntr, lenedg), end="")
        negative_predictions_proba_jcc.append(list(nx.jaccard_coefficient(G, [edge]))[0][2])
        negative_predictions_proba_ra.append(list(nx.resource_allocation_index(G, [edge]))[0][2])
        negative_predictions_proba_aa.append(list(nx.adamic_adar_index(G, [edge]))[0][2])
        negative_predictions_proba_pa.append(list(nx.preferential_attachment(G, [edge]))[0][2])
        negative_predictions_proba_cnsh.append(list(nx.cn_soundarajan_hopcroft(G, [edge]))[0][2]) # needs community information
        negative_predictions_proba_rash.append(list(nx.ra_index_soundarajan_hopcroft(G, [edge]))[0][2]) # needs community information
        negative_predictions_proba_wic.append(list(nx.within_inter_cluster(G, [edge]))[0][2]) # needs community information
        negative_predictions_proba_slp_DegCent.append(list(SLP_prediction(G, [edge], centrality="DegCent"))[0][2])
        negative_predictions_proba_slp_EigenCent.append(list(SLP_prediction(G, [edge], centrality="EigenCent"))[0][2])
        negative_predictions_proba_slp_ClosenessCent.append(list(SLP_prediction(G, [edge], centrality="ClosenessCent"))[0][2])
        negative_predictions_proba_slp_BetweenCent.append(list(SLP_prediction(G, [edge], centrality="BetweenCent"))[0][2])
        negative_predictions_proba_slp_PageRank.append(list(SLP_prediction(G, [edge], centrality="PageRank"))[0][2])
        negative_predictions_proba_slpc_DegCent.append(list(SLPC_prediction(G, [edge], centrality="DegCent"))[0][2]) # needs community information
        negative_predictions_proba_slpc_EigenCent.append(list(SLPC_prediction(G, [edge], centrality="EigenCent"))[0][2]) # needs community information
        negative_predictions_proba_slpc_ClosenessCent.append(list(SLPC_prediction(G, [edge], centrality="ClosenessCent"))[0][2]) # needs community information
        negative_predictions_proba_slpc_BetweenCent.append(list(SLPC_prediction(G, [edge], centrality="BetweenCent"))[0][2]) # needs community information
        negative_predictions_proba_slpc_PageRank.append(list(SLPC_prediction(G, [edge], centrality="PageRank"))[0][2]) # needs community information


    write_pred_proba_to_file("conf/pred-jcc.csv", positive_predictions_proba_jcc, negative_predictions_proba_jcc)
    write_pred_proba_to_file("conf/pred-ra.csv", positive_predictions_proba_ra, negative_predictions_proba_ra)
    write_pred_proba_to_file("conf/pred-aa.csv", positive_predictions_proba_aa, negative_predictions_proba_aa)
    write_pred_proba_to_file("conf/pred-pa.csv", positive_predictions_proba_pa, negative_predictions_proba_pa)
    write_pred_proba_to_file("conf/pred-cnsh.csv", positive_predictions_proba_cnsh, negative_predictions_proba_cnsh)
    write_pred_proba_to_file("conf/pred-rash.csv", positive_predictions_proba_rash, negative_predictions_proba_rash)
    write_pred_proba_to_file("conf/pred-wic.csv", positive_predictions_proba_wic, negative_predictions_proba_wic)
    write_pred_proba_to_file("conf/pred-slp-DegCent.csv", positive_predictions_proba_slp_DegCent, negative_predictions_proba_slp_DegCent)
    write_pred_proba_to_file("conf/pred-slp-EigenCent.csv", positive_predictions_proba_slp_EigenCent, negative_predictions_proba_slp_EigenCent)
    write_pred_proba_to_file("conf/pred-slp-ClosenessCent.csv", positive_predictions_proba_slp_ClosenessCent, negative_predictions_proba_slp_ClosenessCent)
    write_pred_proba_to_file("conf/pred-slp-BetweenCent.csv", positive_predictions_proba_slp_BetweenCent, negative_predictions_proba_slp_BetweenCent)
    write_pred_proba_to_file("conf/pred-slp-PageRank.csv", positive_predictions_proba_slp_PageRank, negative_predictions_proba_slp_PageRank)
    write_pred_proba_to_file("conf/pred-slpc-DegCent.csv", positive_predictions_proba_slpc_DegCent, negative_predictions_proba_slpc_DegCent)
    write_pred_proba_to_file("conf/pred-slpc-EigenCent.csv", positive_predictions_proba_slpc_EigenCent, negative_predictions_proba_slpc_EigenCent)
    write_pred_proba_to_file("conf/pred-slpc-ClosenessCent.csv", positive_predictions_proba_slpc_ClosenessCent, negative_predictions_proba_slpc_ClosenessCent)
    write_pred_proba_to_file("conf/pred-slpc-BetweenCent.csv", positive_predictions_proba_slpc_BetweenCent, negative_predictions_proba_slpc_BetweenCent)
    write_pred_proba_to_file("conf/pred-slpc-PageRank.csv", positive_predictions_proba_slpc_PageRank, negative_predictions_proba_slpc_PageRank)


#     ## Calculating Performance
#     df.loc[0] = ["jcc"] + get_all_perf_measures(positive_predictions_proba_jcc, negative_predictions_proba_jcc, threshold=0.005)
#     df.loc[1] = ["ra"] + get_all_perf_measures(positive_predictions_proba_ra, negative_predictions_proba_ra, threshold=0.005)
#     df.loc[2] = ["aa"] + get_all_perf_measures(positive_predictions_proba_aa, negative_predictions_proba_aa, threshold=0.005)
#     df.loc[3] = ["pa"] + get_all_perf_measures(positive_predictions_proba_pa, negative_predictions_proba_pa, threshold=0.005)
#     df.loc[4] = ["cnsh"] + get_all_perf_measures(positive_predictions_proba_cnsh, negative_predictions_proba_cnsh, threshold=0.005)
#     df.loc[5] = ["rash"] + get_all_perf_measures(positive_predictions_proba_rash, negative_predictions_proba_rash, threshold=0.005)
#     df.loc[6] = ["wic"] + get_all_perf_measures(positive_predictions_proba_wic, negative_predictions_proba_wic, threshold=0.005)
#     df.loc[7] = ["slp"] + get_all_perf_measures(positive_predictions_proba_slp, negative_predictions_proba_slp, threshold=0.0015)
#     df.loc[8] = ["slpc"] + get_all_perf_measures(positive_predictions_proba_slpc, negative_predictions_proba_slpc, threshold=0.0015)
#
#     all_folds_df.append(df)
#
#
# final_df = pd.concat(all_folds_df).groupby(by="algorithm").mean()
# # final_df = all_folds_df[0]
# # col0 = list(all_folds_df[0]["algorithm"])
# print("\n\n")
# print(final_df)
# # final_df.insert(0, "algorithm", col0)
# final_df.to_csv("d:/finaltopol.csv", sep=",")
#
# # print("positive_predictions_proba")
# # print(*positive_predictions_proba[:100], sep="\n")
# # print("\n\nnegative_predictions_proba")
# # print(*negative_predictions_proba[:100], sep="\n")














#
