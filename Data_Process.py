import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import pdb
import os
import pandas as pd
import random

import glob
import os
import pdb
from PIL import Image
import csv

end_set = [600,170,51]

for w in range (1, 4):
    for j in range (1,end_set[w-1]+1):
        y = pd.read_csv('/Users/rqzhang/Desktop/Pearl/Input_Raman_New/'+ str(w) + '_' + str(j) + '.csv', sep=',', header=None)
        y = np.array(y)
        wl = y[:,0]
        count = y[:,1]
        wl = np.flip(wl)
        count = np.flip(count)

        new_wls = np.linspace(102,1999,250)
        new_count = []
        
        for new_wl in new_wls:
            for i in range(len(wl)-1):
                single_wl = wl[i]
                # print(single_wl, new_wl, wl[i+1])
                if single_wl == new_wl:
                    new_count.append(count[i])
                    break
                elif wl[i] < new_wl and wl[i+1] > new_wl:
                    count_left = count[i]
                    count_right = count[i+1]
                    value = (count_right - count_left) / (wl[i+1] - wl[i]) * (new_wl - single_wl) + count_left
                    new_count.append(value)
                    break
        # np.savetxt('/Users/rqzhang/Desktop/Pearl/Input_Raman_Processed_New/'+ str(w) + '_' + str(j) + '.csv', new_count, delimiter=",")
