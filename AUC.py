# Reference: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py

def tied_rank(x):
    """
    Computes the tied rank of elements in x.
    This function computes the tied rank of elements in x.
    Parameters
    ----------
    x : list of numbers, numpy array
    Returns
    -------
    score : list of numbers
            The tied rank f each element in x
    """
    sorted_x = sorted(zip(x,range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_x)-1:
            for j in range(last_rank, i+1):
                r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
    return r

def auc(actual, posterior):
    """
    Computes the area under the receiver-operater characteristic (AUC)
    This function computes the AUC error metric for binary classification.
    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.
    Returns
    -------
    score : double
            The mean squared error between actual and posterior
    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x==1])
    num_negative = len(actual)-num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
    auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
           (num_negative*num_positive))
    return auc

def read_file(input_path):
    file = open(input_path,"r")
    file_lines = file.readlines()
    output_list = []
    for line in file_lines:
        l2 = line.split(",")[1]
        if "A" in l2:
            output_list.append(1)
        else:
            output_list.append(0)

    return output_list

def append_to_lines(input_path, strs_to_append, separator=","):
    file = open(input_path, "r")
    file_lines = file.readlines()

    new_lines = []
    new_lines.append(str(file_lines[0]).replace("\n","") + ",AUC\n")
    for line,rng in zip(file_lines[1:], strs_to_append):
        new_lines.append(str(line).replace("\n","") + separator + '{:0.3f}'.format(rng) + "\n")
        # print(line)

    with open(input_path,'w') as file:
        file.writelines(new_lines)















#
