import getopt
import sys

'''
This program add community information to edges features
table of a network.
If Source node and Target node of an edge are in a same community
the community score = 1 and 0 otherwise.
'''

opts, args = getopt.getopt(sys.argv[1:], "hs:l:f:e:", ["step=", "level="])
print(opts)
print(args)
