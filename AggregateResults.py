import pandas as pd
import os
import sys

jour_conf = sys.argv[1]
steps = sys.argv[2]
preds = sys.argv[3]
pred_year = sys.argv[4]

# file_paths = os.walk(input("Please enter the root directory path for results\n   >>> ").replace("\"", "").replace("\\", "/"))
output_file_path = "E:/Research/Dataset/IEEE Dataset/Final Results/NewApproach/Machine Learning/{}/{}-{}s-{}.csv".format(jour_conf, preds, steps, pred_year)
file_paths = os.walk("E:/Research/Dataset/IEEE Dataset/Final Results/NewApproach/Machine Learning/{}/{}s/{}/".format(jour_conf, steps, preds))
file_paths = list(file_paths)[0]
dir_path = file_paths[0]
file_names = file_paths[2]

dir_path += "/" if dir_path[-1] != "/" else ""


# Choose the final snapshot of the network.
# Consider you want to predict the links which will be appeared in the next 5 years
# then assume that the final snapshot of the network you have is the year 2015,
# so all features will be extracted from nodes that present in the year 2010 for all links
# which appeared in 5 years later (2011-2015). Thus, the train and test dataset is comparised from
# these links.
# Here you should enter the final year of your snapshot
# selected_year = input("Enter the year of prediction:  ")
selected_year = pred_year

selected_file_names = []
for file_name in file_names:
    if file_name.split(".")[0][-2:] == selected_year and file_name.split(".")[1] == "csv":
        selected_file_names.append(file_name)

try:
    selected_file_names = selected_file_names[:4] # I only used 4 iterations
    print("Len Selected files: {}".format(len(selected_file_names)))
    dataFrames = [pd.read_csv("{}{}".format(dir_path, file_name)) for file_name in selected_file_names]

    concatenated_dfs = pd.concat(dataFrames)

    concatenated_dfs = concatenated_dfs.groupby(concatenated_dfs.Classifier).mean()

    # output_file_path = input("Enter the output file path\n   >>> ")
    # print(concatenated_dfs)

    concatenated_dfs.to_csv(output_file_path)
except Exception as e:
    print("No files found for the selected year.")













    #
