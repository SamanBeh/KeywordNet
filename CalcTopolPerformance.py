import pandas as pd
import sys
import random

# sys.argv[1] --> steps
# sys.argv[2] --> jour or conf
# sys.argv[3] --> last year in dataset
# sys.arvg[4] --> column name

column_names = ['Source', 'Target', 'JaccCoeff', 'RACoeff', 'KNLP_DegCent',
'KNLP_EigenCent', 'KNLP_ClosenessCent', 'KNLP_BetweenCent',
'KNLP_PageRank', 'KNLPC_DegCent', 'KNLPC_EigenCent',
'KNLPC_ClosenessCent', 'KNLPC_BetweenCent', 'KNLPC_PageRank']

# column_names = ["Source", "Target", 'comm_gephi']

last_year = sys.argv[3]
first_year = int(sys.argv[3]) - int(sys.argv[1])

pos_path = "E:/Research/Dataset/IEEE Dataset/Final Results/NewApproach/Machine Learning/{}/Features/{}s/features/diff-features-{}-{}.csv".format(sys.argv[2], sys.argv[1], first_year, last_year)
neg_path = "E:/Research/Dataset/IEEE Dataset/Final Results/NewApproach/Machine Learning/{}/Features/{}s/features/miss-features-{}-{}.csv".format(sys.argv[2], sys.argv[1], first_year, last_year)
# col_name = sys.argv[4]
output_path = "E:/Research/Dataset/IEEE Dataset/Final Results/NewApproach/Machine Learning/{}/Features/performance-{}s-{}-{}.csv".format(sys.argv[2], sys.argv[1], first_year, last_year)

pos_data = pd.read_csv(pos_path)
neg_data = pd.read_csv(neg_path)

pos_data = pos_data[column_names]
neg_data = neg_data[column_names]

pos_edges = {(row[1]["Source"], row[1]["Target"]) for row in pos_data.iterrows()}
# neg_edges = {(row[1]["Source"], row[1]["Target"]) for row in neg_data.iterrows()}

pos_neg_data = pos_data.append(neg_data, ignore_index = True)
n_samples_for_AUC = 20000
output_df = pd.DataFrame(columns=["method", "auc", "precision"])
for col_name in column_names[2:]:
    print("Calculating Performance of {}".format(col_name))
    pos_neg_data = pos_neg_data.sort_values(by=[col_name], ascending = False)

    print(" Calculating Precision . . .")
    selected_data = pos_neg_data.iloc[:len(pos_data)]
    selected_edges = [(row[1]["Source"], row[1]["Target"]) for row in selected_data.iterrows()]

    cntr = 0
    for edge in selected_edges:
        if edge in pos_edges:
            cntr += 1

    precision = cntr/len(pos_data)
    print("    {} | Precision: {:0.2f}".format(col_name, precision))

    print(" Calculating AUC . . .")
    all_auc = []
    for i in range(5):
        print("\r  iteration {} of {}".format(i+1, 5), end="")
        cntr1 = 0
        cntr2 = 0
        pos_scores = list(pos_data[col_name])
        neg_scores = list(neg_data[col_name])
        for j in range(n_samples_for_AUC):
            # print("\r   {}/{}".format(j+1, n_samples_for_AUC), end="")
            pos_score = float(random.choice(pos_scores))
            neg_score = float(random.choice(neg_scores))
            if pos_score > neg_score:
                cntr1 += 1
            elif pos_score == neg_score:
                cntr2 += 1

        auc = (cntr1 + cntr2/2) / n_samples_for_AUC
        all_auc.append(auc)
        # print()
        auc_avg = sum(all_auc)/len(all_auc)
    print("    {} | AUC: {:0.2f}\n".format(col_name, auc_avg))
    output_df = output_df.append({"method": col_name, "auc": auc_avg, "precision": precision}, ignore_index = True)


output_df = round(output_df, 2)

output_df.to_csv(output_path, index=False)












#
