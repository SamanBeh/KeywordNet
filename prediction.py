import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle
from datetime import datetime
import math
import os
# from joblib import dump,load

print_log = True
log_file_name = "out/prediction-results-10sj.txt"
classifiers = ["svm", "dtree", "knn", "mnb", "mlp", "rf", "adaboost", "gnb", "qda", "ensemble"]
# classifiers = ["svm", "mlp", "rf", "adaboost", "qda", "ensemble"]
# classifiers = ["mnb"]
# algorithm = "nb"
#########################################################
#########################################################
#########################################################
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

def save_model_pickle(model, root_dir, file_name):
    with open(root_dir + file_name, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model_pickle(model_file_path):
    model = None
    with open(model_file_path, 'rb') as handle:
        model = pickle.load(handle)
    return model

def save_model_joblib(model, root_dir, file_name):
    dump(model, root_dir + file_name)


def perform_prediction(model, pred_pos, pred_neg, print_log=False):
    keys = ["tp","tn","fp","fn","sensitivity","specificity","precision","npv","fnr","fpr","fdr","forate","acc","f1score","mcc", "auc"]
    prediction_result_dict = dict.fromkeys(keys, 0.0)

    # model = None
    # if algorithm.lower() in classifiers:
    #     if algorithm.lower() == "svm":
    #         model = svc
    #     if algorithm.lower() == "dtree":
    #         model = dtree
    #     if algorithm.lower() == "knn":
    #         model = knn
    #     if algorithm.lower() == "mnb":
    #         model = nb
    #     if algorithm.lower() == "mlp":
    #         model = mlp

    # print("-------------------------")
    # positive
    pos = []
    for ind in range(len(pred_pos)):
        pos.append(model.predict([list(pred_pos.iloc[ind])]))

    TP = (pos.count(1)*100)/len(pred_pos)
    FP = 100 - TP
    tp = " TP:                                     {:0.2f}".format(TP)
    fp = " FP:                                     {:0.2f}".format(FP)
    # print(model.predict([[1000866.0, 1004412.0, 73.0, 308.0, -0.19047619047619047, -0.06176412026282601, 0.9619047619047618, 0.3250906633061285, 9.180896135650382e-05, 0.003146519722743909, 0.0010527774199952436, 0.008937373072008988, 8.618557995639522e-05, 0.0003531808078744568, 0.00010284442871768538, 0.0022685872759771212, 0.020893547315568584, 0.014465371691336102, 0.3435217544774928, 0.4143841540638441, 6.108615876405912e-08, 1.1217872216326463e-05, 1.2217231752811823e-08, 1.9550706466753154e-05, -4.913060445975555e-06, 1.7254143396116048e-05, 3.648023930668757e-05, 8.756506027773786e-05]]))


    # negative
    neg = []
    for ind in range(len(pred_neg)):
        neg.append(model.predict([list(pred_neg.iloc[ind])]))
    TN = (neg.count(0)*100)/len(pred_neg)
    FN = 100 - TN
    tn = " TN:                                     {:0.2f}".format(TN)
    fn = " FN:                                     {:0.2f}".format(FN)
    # print(model.predict([[1002786.0, 1004908.0, 161.0, 30.0, 0.07503254207408311, 0.039080459770114984, 0.4770200837485915, 0.55816091954023, 0.006931761474782368, -0.0002556987176820956, 0.01817537832011209, 0.004083275558305315, 0.0020802175984303267, -0.0003483306599907541, 0.008310175122410028, 0.001459972713427101, 0.019676984208538528, 0.006209392933536007, 0.4508021415971623, 0.3942441428573377, 1.421806659504688e-06, -3.877031175364802e-07, 1.6332054389553976e-05, 8.397393370760852e-07, 1.0472264201645998e-05, -7.624431147669698e-06, 0.00010865696771750338, 4.197451598325426e-05]]))

    sensitivity = " Sensitivity or Recall:                  {:0.2f}".format(calculate_performance("sensitivity", TP, TN, FP, FN))
    specificity = " Specificity or Selectivity:             {:0.2f}".format(calculate_performance("specificity", TP, TN, FP, FN))
    precision = " precision or positive predictive value: {:0.2f}".format(calculate_performance("precision", TP, TN, FP, FN))
    npv = " negative predictive value:              {:0.2f}".format(calculate_performance("npv", TP, TN, FP, FN))
    fnr = " miss rate or false negative rate:       {:0.2f}".format(calculate_performance("fnr", TP, TN, FP, FN))
    fpr = " fall-out or false positive rate:        {:0.2f}".format(calculate_performance("fpr", TP, TN, FP, FN))
    fdr = " false discovery rate:                   {:0.2f}".format(calculate_performance("fdr", TP, TN, FP, FN))
    forate = " false omission rate:                    {:0.2f}".format(calculate_performance("forate", TP, TN, FP, FN))
    acc = " Accuracy:                               {:0.2f}".format(calculate_performance("acc", TP, TN, FP, FN))
    f1score = " F1 Score:                               {:0.2f}".format(calculate_performance("f1score", TP, TN, FP, FN))
    mcc = " Matthews correlation coefficient:       {:0.2f}".format(calculate_performance("mcc", TP, TN, FP, FN))

    from AUC import auc
    auc = " Area Under the Curve:                   {:0.2f}".format(auc([1 for i in range(len(pred_pos))] + [0 for i in range(len(pred_neg))], pos + neg))



    prediction_result_dict["tp"] = float(tp.split(":")[1].strip())
    prediction_result_dict["tn"] = float(tn.split(":")[1].strip())
    prediction_result_dict["fp"] = float(fp.split(":")[1].strip())
    prediction_result_dict["fn"] = float(fn.split(":")[1].strip())
    prediction_result_dict["sensitivity"] = float(sensitivity.split(":")[1].strip())
    prediction_result_dict["specificity"] = float(specificity.split(":")[1].strip())
    prediction_result_dict["precision"] = float(precision.split(":")[1].strip())
    prediction_result_dict["npv"] = float(npv.split(":")[1].strip())
    prediction_result_dict["fnr"] = float(fnr.split(":")[1].strip())
    prediction_result_dict["fpr"] = float(fpr.split(":")[1].strip())
    prediction_result_dict["fdr"] = float(fdr.split(":")[1].strip())
    prediction_result_dict["forate"] = float(forate.split(":")[1].strip())
    prediction_result_dict["acc"] = float(acc.split(":")[1].strip())
    prediction_result_dict["f1score"] = float(f1score.split(":")[1].strip())
    prediction_result_dict["mcc"] = float(mcc.split(":")[1].strip())
    prediction_result_dict["auc"] = float(auc.split(":")[1].strip())

    if print_log:
        result = ""
        result = result + "\n\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n\n\n".format(tp, tn, fp, fn, sensitivity, specificity, precision, npv, fnr, fpr, fdr, forate, acc, f1score, mcc, auc)
        print(result)
        with open(log_file_name, "a") as file:
            file.write(result)

    return prediction_result_dict

#########################################################
#########################################################
#########################################################




# positive_data = pd.read_csv(input("   Please enter the positive dataset file path: \n    >>> "))
# negative_data = pd.read_csv(input("   Please enter the negative dataset file path: \n    >>> "))
# path_to_save_model = input("   Please enter a path to save model: [e.g. D:/model.pickle]\n    >>> ")
# algorithm = input("   Which calssifier algorithm do you prefer? {}\n    > ".format(classifiers))
positive_data = pd.read_csv("/home/saman/KW/newall/corrs/10Jcorrdiff-WithComm.csv")
negative_data = pd.read_csv("/home/saman/KW/newall/jmiss/MissLnks-10J-WithComm.csv")
# path_to_save_model = "/home/saman/KW/newall/model-5Jcd-{}.pickle".format(algorithm)
# use_comm_info = input("Do you want to use community info [y/n]:  ")
use_comm_info = "y"
# K = int(input("K(number of iterations to test each classifier):  "))
K = 5
# save_models = input("Do you want to save models? [y/n]:  ")
save_models = "y"
root_dir = ""
if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
    # root_dir = input("Enter the root dir to save models:\n   >>>  ")
    root_dir = "models/10j/"

# output_df_path = input("Please enter the output file path:\n   >>> ")
output_df_path = "out/out-allfeatures-10sj.csv"

# output_df = pd.DataFrame(columns=["Classifier", "TP", "FP", "TN", "FN", "Sensitivity or Recall",\
# "Specificity or Selectivity", "precision or positive predictive value", "negative predictive value",\
# "miss rate or false negative rate", "fall-out or false positive rate", "false discovery rate",\
# "false omission rate", "Accuracy", "F1 Score", "Matthews correlation coefficient"])
#
# if len(output_df) > 0:
#     output_df = pd.read_csv(output_df_path)

result = "-----------------  {}  -----------------".format(datetime.now())
result = result + "\n  Positive file: 4Jcorrdiff.csv"

# columns = ['Source', 'Target',\# 'weight',\
# 'S-degree', 'T-degree',\
# 'S-ClustCoefDiff', 'T-ClustCoefDiff', 'S-ClustCoefAvg', 'T-ClustCoefAvg',\
# 'S-DegCentDiff', 'T-DegCentDiff', 'S-DegCentAVG', 'T-DegCentAVG',\
# 'S-EigenCentDiff', 'T-EigenCentDiff', 'S-EigenCentAVG', 'T-EigenCentAVG',\
# 'S-ClosenessCentDiff', 'T-ClosenessCentDiff', 'S-ClosenessCentAVG', 'T-ClosenessCentAVG',\
# 'S-BetweenCentDiff', 'T-BetweenCentDiff', 'S-BetweenCentAVG', 'T-BetweenCentAVG',\
# 'S-PageRankDiff', 'T-PageRankDiff', 'S-PageRankAVG','T-PageRankAVG']

# selected_cols = ['Source', 'Target',\
# 'S-degree', 'T-degree',\
# 'S-ClustCoefDiff', 'T-ClustCoefDiff', 'S-ClustCoefAvg', 'T-ClustCoefAvg',\
# 'S-DegCentDiff', 'T-DegCentDiff', 'S-DegCentAVG', 'T-DegCentAVG',\
# 'S-EigenCentDiff', 'T-EigenCentDiff', 'S-EigenCentAVG', 'T-EigenCentAVG',\
# 'S-ClosenessCentDiff', 'T-ClosenessCentDiff', 'S-ClosenessCentAVG', 'T-ClosenessCentAVG',\
# 'S-BetweenCentDiff', 'T-BetweenCentDiff', 'S-BetweenCentAVG', 'T-BetweenCentAVG',\
# 'S-PageRankDiff', 'T-PageRankDiff', 'S-PageRankAVG','T-PageRankAVG']

selected_cols = ['Source', 'Target',\
'S-degree', 'T-degree',\
'S-ClustCoefAvg', 'T-ClustCoefAvg',\
'S-DegCentAVG', 'T-DegCentAVG',\
'S-EigenCentAVG', 'T-EigenCentAVG',\
'S-ClosenessCentAVG', 'T-ClosenessCentAVG',\
'S-BetweenCentAVG', 'T-BetweenCentAVG',\
'S-PageRankAVG','T-PageRankAVG']

if use_comm_info.lower() in ["1", "yes", "y", "yeah", "whatever"]:
    selected_cols = ['Source', 'Target',\
    'S-degree', 'T-degree',\
    'S-ClustCoefAvg', 'T-ClustCoefAvg',\
    'S-DegCentAVG', 'T-DegCentAVG',\
    'S-EigenCentAVG', 'T-EigenCentAVG',\
    'S-ClosenessCentAVG', 'T-ClosenessCentAVG',\
    'S-BetweenCentAVG', 'T-BetweenCentAVG',\
    'S-PageRankAVG','T-PageRankAVG', 'comm_gephi']

positive_data_main = positive_data[selected_cols]
negative_data_main = negative_data[selected_cols]

result = result + "\n\n FEATURES [{}]: {}\n".format(len(selected_cols), selected_cols)
print("\n\n FEATURES [{}]: {}\n".format(len(selected_cols), selected_cols))
print("\n\n CLASSIFIERS: {}\n".format(classifiers))

# try:
#     del positive_data["weight"]
# except Exception as e:
#     pass
#
# try:
#     del negative_data["weight"]
# except Exception as e:
#     pass


train_cut_percent = 75 # the rest is the test data
train_cut_count  = int((len(positive_data) * train_cut_percent) / 100)
# train_cut_count  = 700

with open(log_file_name, "w") as file:
    file.write(" # output file path: {}\n".format(output_df_path))
    file.write(" # root dir to save models: {}\n".format(root_dir))
    file.write(" # train cut [percent]: {}\n".format(train_cut_percent))
    file.write(" # SELECTED FEATURES [{}]: {}\n".format(len(selected_cols), selected_cols))
    file.write(" # SELECTED CLASSIFIERS: {}\n+++++++++++++++++++++++++++++++++\n".format(classifiers))

svm_dict_avg = None
dtree_dict_avg = None
knn2_dict_avg = None
knn3_dict_avg = None
knn4_dict_avg = None
knn5_dict_avg = None
knn6_dict_avg = None
nb_dict_avg = None
mlp_dict_avg = None
rfc_dict_avg = None
adaboost_dict_avg = None
gnb_dict_avg = None
qda_dict_avg = None
vc_dict_avg = None

# for algorithm in classifiers:
#     print("  - - - Classifier: {} - - -   ".format(algorithm))

ref_dict = {}
for i in range(K):
    with open(log_file_name, "a") as file:
        file.write("\n######################  STEP [ {} ] STARTED  ######################\n".format(i+1))
    svm_dict = None
    dtree_dict = None
    knn2_dict = None
    knn3_dict = None
    knn4_dict = None
    knn5_dict = None
    knn6_dict = None
    nb_dict = None
    mlp_dict = None
    rfc_dict = None
    adaboost_dict = None
    gnb_dict = None
    qda_dict = None
    vc_dict = None
    print("  Step {}/{}".format(i+1, K))
    print("    shuffling data . . .")
    shuffle(positive_data_main)
    shuffle(negative_data_main)

    # Test Data
    pred_pos = positive_data_main.iloc[train_cut_count:,:]
    pred_neg = negative_data_main.iloc[train_cut_count:len(positive_data_main),:]

    # Train Data
    positive_data = positive_data_main.iloc[:train_cut_count,:]
    negative_data = negative_data_main.iloc[:train_cut_count,:]

    # result = result + "\n  TRAIN [{}%]: len_positive: {} | len_negative: {}".format(train_cut_percent, len(positive_data), len(negative_data))
    # result = result + "\n  TEST  [{}%]: len_positive: {} | len_negative: {}".format(100 - train_cut_percent, len(pred_pos), len(pred_neg))
    # print("\n  TRAIN [{}%]: len_positive: {} | len_negative: {}".format(train_cut_percent, len(positive_data), len(negative_data)))
    # print("  TEST  [{}%]: len_positive: {} | len_negative: {}".format(100 - train_cut_percent, len(pred_pos), len(pred_neg)))

    edges_features = []
    is_connected = []

    edges_features.extend([list(positive_data.iloc[i]) for i in range(len(positive_data))])
    is_connected.extend([1 for i in range(len(positive_data))])

    edges_features.extend([list(negative_data.iloc[i]) for i in range(len(negative_data))])
    is_connected.extend([0 for i in range(len(negative_data))])


    X = np.array(edges_features)
    y = np.array(is_connected)

    for algorithm in classifiers:
        ###########################################
        ########            SVM         ###########
        ###########################################
        if algorithm.lower() == "svm":
            start_time = datetime.now()
            print("   Selected Classifier: SVM")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: SVM")
            result = result + "\n  - - - Selected Classifier: SVM"
            from sklearn import svm, datasets
            C = 1
            svc = None
            if not os.path.isfile(root_dir + "svc-{}.pickle".format(i)):
                svc = svm.SVC(kernel="linear", C=C, max_iter=20000, probability=True).fit(X, y)
            else:
                print(" Loading saved model . . .")
                svc = load_model_pickle(root_dir + "svc-{}.pickle".format(i))
            svm_dict = perform_prediction(model = svc, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                svm_dict_avg = svm_dict
            else:
                for k,n in zip(svm_dict_avg.keys(), svm_dict.keys()):
                    svm_dict_avg[k] = float(svm_dict_avg[k]) + float(svm_dict[k])

            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(svc, root_dir, file_name="svc-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["svm"] = [svm_dict_avg, "svm_dict_avg"]
        ###########################################
        #######            Dtree         ##########
        ###########################################
        if algorithm.lower() == "dtree":
            start_time = datetime.now()
            print("   Selected Classifier: Decision Tree")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: Decision Tree")
            result = result + "\n  - - - Selected Classifier: Decision Tree"
            from sklearn import tree
            dtree = None
            if not os.path.isfile(root_dir + "dtree-{}.pickle".format(i)):
                dtree = tree.DecisionTreeClassifier().fit(X, y)
            else:
                print(" Loading saved model . . .")
                dtree = load_model_pickle(root_dir + "dtree-{}.pickle".format(i))
            dtree_dict = perform_prediction(model = dtree, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                dtree_dict_avg = dtree_dict
            else:
                for k,n in zip(dtree_dict_avg.keys(), dtree_dict.keys()):
                    dtree_dict_avg[k] = float(dtree_dict_avg[k]) + float(dtree_dict[k])

            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(dtree, root_dir, file_name="dtree-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["dtree"] = [dtree_dict_avg, "dtree_dict_avg"]
        ###########################################
        ########            KNN         ###########
        ###########################################
        if algorithm.lower() == "knn":
            for j in range(2,7):
                # k = int(input("   Please enter K = "))
                k = j
                print("   Selected Classifier: K-Nearest Neighbours [k={}]".format(k))
                with open(log_file_name, "a") as file:
                    file.write("\n   Selected Classifier: K-Nearest Neighbours [k={}]".format(k))
                result = result + "\n  - - - Selected Classifier: K-Nearest Neighbours [k={}]".format(k)
                from sklearn.neighbors import KNeighborsClassifier
                knn = None
                if not os.path.isfile(root_dir + "knn{}-{}.pickle".format(j, i)):
                    knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
                else:
                    print(" Loading saved model . . .")
                    knn = load_model_pickle(root_dir + "knn{}-{}.pickle".format(j, i))
                if j == 2:
                    start_time = datetime.now()
                    knn2_dict = perform_prediction(model = knn, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
                    if i == 0:
                        knn2_dict_avg = knn2_dict
                    else:
                        for k,n in zip(knn2_dict_avg.keys(), knn2_dict.keys()):
                            knn2_dict_avg[k] = float(knn2_dict_avg[k]) + float(knn2_dict[k])
                    if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                        save_model_pickle(knn, root_dir, file_name="knn2-{}.pickle".format(i))
                    elapsed_time = datetime.now() - start_time
                    print("\n  * Elapsed time: {}".format(elapsed_time))
                    print("-------------------------")
                    with open(log_file_name, "a") as file:
                        file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
                    ref_dict["knn2"] = [knn2_dict_avg, "knn2_dict_avg"]
                elif j == 3:
                    start_time = datetime.now()
                    knn3_dict = perform_prediction(model = knn, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
                    if i == 0:
                        knn3_dict_avg = knn3_dict
                    else:
                        for k,n in zip(knn3_dict_avg.keys(), knn3_dict.keys()):
                            knn3_dict_avg[k] = float(knn3_dict_avg[k]) + float(knn3_dict[k])
                    if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                        save_model_pickle(knn, root_dir, file_name="knn3-{}.pickle".format(i))
                    elapsed_time = datetime.now() - start_time
                    print("\n  * Elapsed time: {}".format(elapsed_time))
                    print("-------------------------")
                    with open(log_file_name, "a") as file:
                        file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
                    ref_dict["knn3"] = [knn3_dict_avg, "knn3_dict_avg"]
                elif j == 4:
                    start_time = datetime.now()
                    knn4_dict = perform_prediction(model = knn, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
                    if i == 0:
                        knn4_dict_avg = knn4_dict
                    else:
                        for k,n in zip(knn4_dict_avg.keys(), knn4_dict.keys()):
                            knn4_dict_avg[k] = float(knn4_dict_avg[k]) + float(knn4_dict[k])
                    if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                        save_model_pickle(knn, root_dir, file_name="knn4-{}.pickle".format(i))
                    elapsed_time = datetime.now() - start_time
                    print("\n  * Elapsed time: {}".format(elapsed_time))
                    print("-------------------------")
                    with open(log_file_name, "a") as file:
                        file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
                    ref_dict["knn4"] = [knn4_dict_avg, "knn4_dict_avg"]
                elif j == 5:
                    start_time = datetime.now()
                    knn5_dict = perform_prediction(model = knn, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
                    if i == 0:
                        knn5_dict_avg = knn5_dict
                    else:
                        for k,n in zip(knn5_dict_avg.keys(), knn5_dict.keys()):
                            knn5_dict_avg[k] = float(knn5_dict_avg[k]) + float(knn5_dict[k])
                    if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                        save_model_pickle(knn, root_dir, file_name="knn5-{}.pickle".format(i))
                    elapsed_time = datetime.now() - start_time
                    print("\n  * Elapsed time: {}".format(elapsed_time))
                    print("-------------------------")
                    with open(log_file_name, "a") as file:
                        file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
                    ref_dict["knn5"] = [knn5_dict_avg, "knn5_dict_avg"]
                elif j == 6:
                    start_time = datetime.now()
                    knn6_dict = perform_prediction(model = knn, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
                    if i == 0:
                        knn6_dict_avg = knn6_dict
                    else:
                        for k,n in zip(knn6_dict_avg.keys(), knn6_dict.keys()):
                            knn6_dict_avg[k] = float(knn6_dict_avg[k]) + float(knn6_dict[k])
                    if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                        save_model_pickle(knn, root_dir, file_name="knn6-{}.pickle".format(i))
                    elapsed_time = datetime.now() - start_time
                    print("\n  * Elapsed time: {}".format(elapsed_time))
                    print("-------------------------")
                    with open(log_file_name, "a") as file:
                        file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
                    ref_dict["knn6"] = [knn6_dict_avg, "knn6_dict_avg"]
        ###########################################
        ########            MNB         ###########
        ###########################################
        if algorithm.lower() == "mnb":
            start_time = datetime.now()
            print("   Selected Classifier: Multinomial Naive Bayes")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: Multinomial Naive Bayes")
            result = result + "\n  - - - Selected Classifier: Multinomial Naive Bayes"
            from sklearn.naive_bayes import MultinomialNB
            nb = None
            if not os.path.isfile(root_dir + "nb-{}.pickle".format(i)):
                nb = MultinomialNB().fit(X, y)
            else:
                print(" Loading saved model . . .")
                nb = load_model_pickle(root_dir + "nb-{}.pickle".format(i))
            nb_dict = perform_prediction(model = nb, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                nb_dict_avg = nb_dict
            else:
                for k,n in zip(nb_dict_avg.keys(), nb_dict.keys()):
                    nb_dict_avg[k] = float(nb_dict_avg[k]) + float(nb_dict[k])
            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(nb, root_dir, file_name="nb-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["mnb"] = [nb_dict_avg, "nb_dict_avg"]

        ###########################################
        ########            MLP         ###########
        ###########################################
        if algorithm.lower() == "mlp":
            start_time = datetime.now()
            print("   Selected Classifier: Multi-layer Perceptron Classifier | max_iter=50")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: Multi-layer Perceptron Classifier | max_iter=50")
            from sklearn.neural_network import MLPClassifier
            mlp = None
            if not os.path.isfile(root_dir + "mlp-{}.pickle".format(i)):
                mlp = MLPClassifier(alpha=1, max_iter=50).fit(X, y)
            else:
                print(" Loading saved model . . .")
                mlp = load_model_pickle(root_dir + "mlp-{}.pickle".format(i))
            mlp_dict = perform_prediction(model = mlp, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                mlp_dict_avg = mlp_dict
            else:
                for k,n in zip(mlp_dict_avg.keys(), mlp_dict.keys()):
                    mlp_dict_avg[k] = float(mlp_dict_avg[k]) + float(mlp_dict[k])
            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(mlp, root_dir, file_name="mlp-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["mlp"] = [mlp_dict_avg, "mlp_dict_avg"]

        '''
        ###########################################
        ########            MLP         ###########
        ###########################################
        if algorithm.lower() == "mlp":
            start_time = datetime.now()
            print("   Selected Classifier: Multi-layer Perceptron Classifier | max_iter=100")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: Multi-layer Perceptron Classifier | max_iter=100")
            from sklearn.neural_network import MLPClassifier
            mlp = None
            if not os.path.isfile(root_dir + "mlp-{}.pickle".format(i)):
                mlp = MLPClassifier(alpha=1, max_iter=100).fit(X, y)
            else:
                print(" Loading saved model . . .")
                mlp = load_model_pickle(root_dir + "mlp-{}.pickle".format(i))
            mlp_dict = perform_prediction(model = mlp, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                mlp_dict_avg = mlp_dict
            else:
                for k,n in zip(mlp_dict_avg.keys(), mlp_dict.keys()):
                    mlp_dict_avg[k] = float(mlp_dict_avg[k]) + float(mlp_dict[k])
            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(mlp, root_dir, file_name="mlp-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["mlp"] = [mlp_dict_avg, "mlp_dict_avg"]

        ###########################################
        ########            MLP         ###########
        ###########################################
        if algorithm.lower() == "mlp":
            start_time = datetime.now()
            print("   Selected Classifier: Multi-layer Perceptron Classifier | max_iter=500")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: Multi-layer Perceptron Classifier | max_iter=500")
            from sklearn.neural_network import MLPClassifier
            mlp = None
            if not os.path.isfile(root_dir + "mlp-{}.pickle".format(i)):
                mlp = MLPClassifier(alpha=1, max_iter=500).fit(X, y)
            else:
                print(" Loading saved model . . .")
                mlp = load_model_pickle(root_dir + "mlp-{}.pickle".format(i))
            mlp_dict = perform_prediction(model = mlp, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                mlp_dict_avg = mlp_dict
            else:
                for k,n in zip(mlp_dict_avg.keys(), mlp_dict.keys()):
                    mlp_dict_avg[k] = float(mlp_dict_avg[k]) + float(mlp_dict[k])
            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(mlp, root_dir, file_name="mlp-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["mlp"] = [mlp_dict_avg, "mlp_dict_avg"]

        ###########################################
        ########            MLP         ###########
        ###########################################
        if algorithm.lower() == "mlp":
            start_time = datetime.now()
            print("   Selected Classifier: Multi-layer Perceptron Classifier | max_iter=1000")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: Multi-layer Perceptron Classifier | max_iter=1000")
            from sklearn.neural_network import MLPClassifier
            mlp = None
            if not os.path.isfile(root_dir + "mlp-{}.pickle".format(i)):
                mlp = MLPClassifier(alpha=1, max_iter=1000).fit(X, y)
            else:
                print(" Loading saved model . . .")
                mlp = load_model_pickle(root_dir + "mlp-{}.pickle".format(i))
            mlp_dict = perform_prediction(model = mlp, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                mlp_dict_avg = mlp_dict
            else:
                for k,n in zip(mlp_dict_avg.keys(), mlp_dict.keys()):
                    mlp_dict_avg[k] = float(mlp_dict_avg[k]) + float(mlp_dict[k])
            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(mlp, root_dir, file_name="mlp-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["mlp"] = [mlp_dict_avg, "mlp_dict_avg"]
        '''
        ###########################################
        #########            RF         ###########
        ###########################################
        if algorithm.lower() == "rf":
            start_time = datetime.now()
            print("   Selected Classifier: Random Forest Classifier")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: Random Forest Classifier")
            from sklearn.ensemble import RandomForestClassifier
            rfc = None
            if not os.path.isfile(root_dir + "rfc-{}.pickle".format(i)):
                rfc = RandomForestClassifier(max_depth = None, n_estimators=200).fit(X, y)
            else:
                print(" Loading saved model . . .")
                rfc = load_model_pickle(root_dir + "rfc-{}.pickle".format(i))
            rfc_dict = perform_prediction(model = rfc, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                rfc_dict_avg = rfc_dict
            else:
                for k,n in zip(rfc_dict_avg.keys(), rfc_dict.keys()):
                    rfc_dict_avg[k] = float(rfc_dict_avg[k]) + float(rfc_dict[k])
            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(rfc, root_dir, file_name="rfc-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["rf"] = [rfc_dict_avg, "rfc_dict_avg"]

        ###########################################
        ######           adaboost         #########
        ###########################################
        if algorithm.lower() == "adaboost":
            start_time = datetime.now()
            print("   Selected Classifier: AdaBoost Classifier")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: AdaBoost Classifier")
            from sklearn.ensemble import AdaBoostClassifier
            adaboost = None
            if not os.path.isfile(root_dir + "adaboost-{}.pickle".format(i)):
                adaboost = AdaBoostClassifier(n_estimators=100).fit(X, y)
            else:
                print(" Loading saved model . . .")
                adaboost = load_model_pickle(root_dir + "adaboost-{}.pickle".format(i))
            adaboost_dict = perform_prediction(model = adaboost, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                adaboost_dict_avg = adaboost_dict
            else:
                for k,n in zip(adaboost_dict_avg.keys(), adaboost_dict.keys()):
                    adaboost_dict_avg[k] = float(adaboost_dict_avg[k]) + float(adaboost_dict[k])
            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(adaboost, root_dir, file_name="adaboost-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["adaboost"] = [adaboost_dict_avg, "adaboost_dict_avg"]

        ###########################################
        ########            GNB         ###########
        ###########################################
        if algorithm.lower() == "gnb":
            start_time = datetime.now()
            print("   Selected Classifier: Gaussian Naive Bayes")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: Gaussian Naive Bayes")
            from sklearn.naive_bayes import GaussianNB
            gnb = None
            if not os.path.isfile(root_dir + "gnb-{}.pickle".format(i)):
                gnb = GaussianNB().fit(X, y)
            else:
                print(" Loading saved model . . .")
                gnb = load_model_pickle(root_dir + "gnb-{}.pickle".format(i))
            gnb_dict = perform_prediction(model = gnb, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                gnb_dict_avg = gnb_dict
            else:
                for k,n in zip(gnb_dict_avg.keys(), gnb_dict.keys()):
                    gnb_dict_avg[k] = float(gnb_dict_avg[k]) + float(gnb_dict[k])
            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(gnb, root_dir, file_name="gnb-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["gnb"] = [gnb_dict_avg, "gnb_dict_avg"]

        ###########################################
        ########            QDA         ###########
        ###########################################
        if algorithm.lower() == "qda":
            start_time = datetime.now()
            print("   Selected Classifier: Quadratic Discriminant Analysis")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: Quadratic Discriminant Analysis")
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            qda = None
            if not os.path.isfile(root_dir + "qda-{}.pickle".format(i)):
                qda = QuadraticDiscriminantAnalysis().fit(X, y)
            else:
                print(" Loading saved model . . .")
                qda = load_model_pickle(root_dir + "qda-{}.pickle".format(i))
            qda_dict = perform_prediction(model = qda, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                qda_dict_avg = qda_dict
            else:
                for k,n in zip(qda_dict_avg.keys(), qda_dict.keys()):
                    qda_dict_avg[k] = float(qda_dict_avg[k]) + float(qda_dict[k])
            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(qda, root_dir, file_name="qda-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["qda"] = [qda_dict_avg, "qda_dict_avg"]


    ###########################################
    ######          ENSEMBLE           ########
    ###########################################
    if "ensemble" in classifiers:
        if len(classifiers) > 1:
            start_time = datetime.now()
            print("   Selected Classifier: Ensemble Learning / Voting Classifier")
            with open(log_file_name, "a") as file:
                file.write("\n   Selected Classifier: Ensemble Learning / Voting Classifier")
            from sklearn.ensemble import VotingClassifier
            vc = None
            if not os.path.isfile(root_dir + "vc-{}.pickle".format(i)):
                vc = VotingClassifier(estimators=[('svm', svc), ('mlp', mlp), ('rf', rfc), ('adaboost', adaboost), ('qda', qda)], voting='soft', weights=[1,1,1,1,1]).fit(X, y)
            else:
                print(" Loading saved model . . .")
                vc = load_model_pickle(root_dir + "vc-{}.pickle".format(i))
            vc_dict = perform_prediction(model = vc, pred_pos = pred_pos, pred_neg = pred_neg, print_log=print_log)
            if i == 0:
                vc_dict_avg = vc_dict
            else:
                for k,n in zip(vc_dict_avg.keys(), vc_dict.keys()):
                    vc_dict_avg[k] = float(vc_dict_avg[k]) + float(vc_dict[k])
            if save_models.lower() in ["1", "yes", "y", "yeah", "whatever"]:
                save_model_pickle(vc, root_dir, file_name="vc-{}.pickle".format(i))
            elapsed_time = datetime.now() - start_time
            print("\n  * Elapsed time: {}".format(elapsed_time))
            print("-------------------------")
            with open(log_file_name, "a") as file:
                file.write("\n  Elapsed time: {}\n-------------------------".format(elapsed_time))
            ref_dict["ensemble"] = [vc_dict_avg, "vc_dict_avg"]


    with open(log_file_name, "a") as file:
        file.write("\n######################  STEP [ {} ] FINISHED  ######################\n".format(i+1))
    # ref_dict = {"svm": [svm_dict_avg, "svm_dict_avg"], "dtree": [dtree_dict_avg, "dtree_dict_avg"],\
    # "knn3": [knn3_dict_avg, "knn3_dict_avg"], "knn4": [knn4_dict_avg, "knn4_dict_avg"], "knn5": [knn5_dict_avg, "knn5_dict_avg"],\
    # "mnb": [nb_dict_avg, "nb_dict_avg"], "mlp": [mlp_dict_avg, "mlp_dict_avg"], "rf": [rfc_dict_avg, "rfc_dict_avg"],\
    # "adaboost": [adaboost_dict_avg, "adaboost_dict_avg"], "gnb": [gnb_dict_avg, "gnb_dict_avg"],\
    # "qda": [qda_dict_avg, "qda_dict_avg"], "ensemble": [vc_dict_avg, "vc_dict_avg"]}

dcts = []
dcts_names = []
for classifier in ref_dict.keys():
    dcts.append(ref_dict[classifier][0])
    dcts_names.append(ref_dict[classifier][1])

cols = ["Classifier"] + list(dcts[0].keys())
# print("COLUMNS: {}".format(cols))
output_df = pd.DataFrame(columns=cols)


for dct in dcts:
    count = K
    # print("\ncount: {}".format(count))
    keys = list(dct.keys())
    # print("keys: {}".format(keys))
    # print("1st values: {}".format(list(dct.values())))
    for key in keys:
        dct[key] = float(dct[key]) / count
    # print("2nd values: {}".format(list(dct.values())))


for dct, dct_name, i in zip(dcts, dcts_names, range(len(dcts))):
    output_df.loc[i] = [dct_name] + list(dct.values())
    # print("3rd values: {}".format(output_df.loc[i]))


output_df.to_csv(output_df_path, index=False)


















#
