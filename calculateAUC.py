# Calculates AUC for topology based prediction results
from AUC import auc
import pandas as pd
import pyperclip

input_file_path = input("Please enter the file path:\n   >>> ")
threshold = float(input("Please enter the threshold:  "))


df = pd.read_csv(input_file_path, low_memory=False)

pos = [x for x in list(df.positive) if str(x) != "nan"]
neg = [x for x in list(df.negative) if str(x) != "nan"]

predPos = []
for pp in pos:
    if pp >= threshold:
        predPos.append(1)
    else:
        predPos.append(0)

predNeg = []
for pn in neg:
    if pn >= threshold:
        predNeg.append(1)
    else:
        predNeg.append(0)

positives = [1 for x in range(len(predPos))]
negatives = [0 for x in range(len(predNeg))]

auc = auc(positives + negatives, predPos + predNeg)

pyperclip.copy(auc)
print(" Area Under the Curve: {:0.2f}".format(float(pyperclip.paste())))
