import sys, os
filedirectory = os.path.dirname(os.path.realpath(__file__))
rootdirectory = os.path.split(filedirectory)[0]
sys.path.insert(0, rootdirectory)