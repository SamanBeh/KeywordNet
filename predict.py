import subprocess as sp
import KeepAlive



KeepAlive.start_thread()
try:
    proc = sp.Popen("python prediction-svm.py", shell=True)
    proc.wait()
except Exception as e:
    KeepAlive.finish_thread()

KeepAlive.finish_thread()








#
