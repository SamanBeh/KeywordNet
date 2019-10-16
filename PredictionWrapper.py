# This program is to run the Prediction.py multiple times
import sys
import subprocess as sp

steps = int(sys.argv[1])

for year in range(17, 17-5, -1):
    for i in range(10):
        print("STARTED  - python Prediction.py {} {} {}".format(year, steps, i))
        proc = sp.Popen("python Prediction.py {} {} {}".format(year, steps, i), shell=True)
        proc.wait()
        print("FINISHED - python Prediction.py {} {} {}".format(year, steps, i))
