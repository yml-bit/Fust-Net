import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import scipy.io as scio
import random
import pandas as pd
from openpyxl import load_workbook
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def external_list():
    CT_value_file = "./data/DrXiong_files_2"
    external = "external_list.txt"
    f1 = open(external, "w")
    file_list = os.listdir(CT_value_file)
    for q in range(len(file_list)):
        f1.writelines(file_list[q] + "\n")
    f1.close()

def internal_data():
    CT_value_file = "./data/CT_value_20231219"
    file_list = os.listdir(CT_value_file)
    treatment_file = "./data/treatment_para/"
    patient=0
    total=0
    Sonication_duration = []
    Actual_energy = []
    Actual_power = []
    mmt=[]
    for file in file_list:
        path = os.path.join(treatment_file, file)+".csv"
        data = pd.read_csv(path)
        duration = data.iloc[:, 10]
        energy = data.iloc[:, 12]
        power = data.iloc[:, 8]
        mm = data.iloc[:, 25]
        aa = data.iloc[:, 19]
        # print(aa[1])
        indices_of_no = aa[aa == 'No             '].index
        Sonication_duration.extend(duration[indices_of_no])
        Actual_energy.extend(energy[indices_of_no])
        Actual_power.extend(power[indices_of_no])
        mmt.extend(mm[indices_of_no])
        if len(indices_of_no) == len(aa):
            patient += 1
        total += len(indices_of_no)
    print("effective patients:%.2f" % patient)
    print("effective number:%.2f" % total)

    print("mean maximal temperature mean:%.2f" % np.mean(mmt))
    print("mean maximal temperature std:%.2f" % np.std(mmt, ddof=1))
    print("Sonication_duration mean:%.2f" % np.mean(Sonication_duration))
    print("Sonication_duration std:%.2f" % np.std(Sonication_duration, ddof=1))

    print("Actual_energy mean:%.2f" % np.mean(Actual_energy))
    print("Actual_energy std:%.2f" % np.std(Actual_energy, ddof=1))

    print("Actual_power mean:%.2f" % np.mean(Actual_power))
    print("Actual_power std:%.2f" % np.std(Actual_power, ddof=1))

def external_data():
    treatment_file = "./data/TreatSummary/"
    file_list = os.listdir(treatment_file)
    patient=0
    total=0
    Sonication_duration=[]
    Actual_energy=[]
    Actual_power=[]
    mmt=[]
    for file in file_list:
        path = os.path.join(treatment_file, file) + "/TreatSummary.csv"
        data = pd.read_csv(path)
        duration = data.iloc[:, 10]
        energy = data.iloc[:, 12]
        power= data.iloc[:, 8]
        mm = data.iloc[:, 25]
        aa=data.iloc[:, 19]
        # print(aa[1])
        indices_of_no = aa[aa =='No             '].index
        Sonication_duration.extend(duration[indices_of_no])
        Actual_energy.extend(energy[indices_of_no])
        Actual_power.extend(power[indices_of_no])
        mmt.extend(mm[indices_of_no])
        if len(indices_of_no)==len(aa):
            patient+=1
        total+=len(indices_of_no)
    print("effective patients:%.2f"%patient)
    print("effective number:%.2f"%total)

    print("mean maximal temperature mean:%.2f"%np.mean(mmt))
    print("mean maximal temperature std:%.2f" % np.std(mmt,ddof=1))

    print("Sonication_duration mean:%.2f"%np.mean(Sonication_duration))
    print("Sonication_duration std:%.2f" % np.std(Sonication_duration,ddof=1))

    print("Actual_energy mean:%.2f"%np.mean(Actual_energy))
    print("Actual_energy std:%.2f" % np.std(Actual_energy,ddof=1))

    print("Actual_power mean:%.2f"%np.mean(Actual_power))
    print("Actual_power std:%.2f" % np.std(Actual_power,ddof=1))

if __name__ == '__main__':
    internal_data()
    external_data()