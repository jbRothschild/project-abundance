import numpy as np; import csv; import os
import matplotlib.pyplot as plt
from gillespie_models import MultiLV, RESULTS_DIR

sim_dir = "multiLV0"
dict_sim = {}
# get parameters of simulation
with open(os.getcwd() + os.sep + RESULTS_DIR + os.sep + sim_dir + os.sep + "params.csv", newline="") as paramfile:
    reader = csv.reader(paramfile)
    next(reader)
    dict_sim = dict(reader)

# turn certain strings to numbers
for i in dict_sim:
    if i != 'sim_dir':
        dict_sim[i] = float(dict_sim[i])

Model = MultiLV(**dict_sim)
Model.analysis()
