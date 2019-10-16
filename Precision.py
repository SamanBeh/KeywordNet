import random
import pandas as pd
import GenerateMissingLinks
from random import shuffle


def calculate_precision(number_of_iterations, old_edges_file_path, new_edges_file_path, num_of_random_non_edges):
    '''
    Precision for link prediction:
    Precision is the ratio of correctly predicted items selected to the
    number of items selected.

    Precision = Lr/L

    which L is the number of all edges in our sample set and
    Lr is the number of edges that predicted correctly
    '''
    for i in range(number_of_iterations):
        otpt = get_missing_links(old_edges_file_path, new_edges_file_path, output_file_path, num_of_edges)
        positive_edges = otpt["new_edges"]
        negative_edges = otpt["missing_edges"]
        samples = list(positive_edges) + list(negative_edges)
        shuffle(samples)


        for edge in samples[:len(positive_edges)]:
            #TODO: ...





num_of_random_non_edges = input("   number of negative edges you want to create:  ")
if num_of_random_non_edges == None or num_of_random_non_edges == "":
    num_of_random_non_edges = 100000
else:
    try:
        num_of_random_non_edges = int(num_of_random_non_edges)
    except Exception as e:
        print(e)

# i.e. edges which we have until 2016
old_edges_file_path = input("   Please enter OLD edges file path1: \n    >>> ")
# i.e. edges which appeared in 2017
new_edges_file_path = input("   Please enter NEW edges file path1: \n    >>> ")
# positive_dataset = input("   Please enter positive dataset of edges: \n   >>> ")
output_file_path = input("   Please enter the output file path1: \n    >>> ")
