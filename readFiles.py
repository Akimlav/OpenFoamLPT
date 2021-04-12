#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:43:04 2021

@author: akimlavrinenko
"""

import numpy as np
import vtk
import matplotlib.pyplot as plt
from NEK5000_func_lib import fast_scandir, listfile, find_in_list_of_list


dirpath = '/Users/akimlavrinenko/Documents/coding/data/OF_ps_data/test0_k0'
fold_name = 'posititions_k0'


folders = fast_scandir(dirpath)
folders = [word for word in folders if fold_name in word]
folders.sort()

listOfFileList, allFileList = listfile(folders, '.vtk')

#params
step = 1 # file step
allFileList = allFileList[0::step]
allFileList.sort()
allFileList = allFileList[:32]
allFileList.sort()

def vtkReader(file):
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(file)
    reader.Update()
    points = np.array( reader.GetOutput().GetPoints().GetData() )
    return points

meanX = []
meanZ = []
for file in allFileList:
    ind = find_in_list_of_list(listOfFileList, file)
    if file in listOfFileList[ind[0]]:
        path = folders[ind[0]] + '/'
        data = vtkReader(path + file)
        coords = np.asarray(data)
        # print(len(coords), file)
        x = coords[:,][:,0]
        z = coords[:,][:,2]
        xm = np.mean(x)
        zm = np.mean(z)
        meanX.append(xm)
        meanZ.append(zm)
        

#DNS data
positionOrig=np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/DNS_cough/s1e1_xc.dat', invalid_raise = False)
positionX4 = positionOrig[:,3] * 0.02
positionY4 = positionOrig[:,2] * 0.02
#end of DNS data

meanX[0] = 0
meanZ[0] = 0
fig, axs = plt.subplots(figsize=(5, 5))
# fig, ax = plt.subplots()
# ax.grid(True)
plt.plot(meanX, meanZ,'-',label="RANS 4e-6 m")
plt.plot(positionX4, positionY4,'-',label="DNS 4e-6 m")
plt.xlabel('x')
plt.ylabel('z')
plt.ylim(-0.05, 0.05)
plt.xlim(0,0.6)
plt.grid()
plt.legend(loc="upper right")
plt.savefig('DNS_RANS_mean_disctance.png', dpi= 150)
plt.show()




