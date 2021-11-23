#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 19:50:25 2021

@author: akimlavrinenko
"""
from openFoamClass import openFoam
import numpy as np

# dirPath = '/Users/akimlavrinenko/Documents/coding/data/cough/RANS_cough_data/pure_data/VTK/lagrangian'
dirPath = '/Users/akimlavrinenko/Documents/coding/data/cough/RANS_cough_data/pure_data/VTK/lagrangian'

foldName = 'reactingCloud1'
fileName = '10kk_pData'

folders = openFoam().fast_scandir(dirPath)
folders = [word for word in folders if foldName in word]
folders.sort()
listOfFileList, allFileList = openFoam().listfile(folders, '.vtk')
allFileList.sort()

timeStep = 0.025
t = np.linspace(1*timeStep ,((len(allFileList))*timeStep),num = len(allFileList))
# data = [origId, d, T, vec(U), points(x,y,z)]

resList = []
for file in allFileList:
    meanData = []
    ind = openFoam().find_in_list_of_list(listOfFileList, file)
    if file in listOfFileList[ind[0]]:
        path = folders[ind[0]] + '/'
        data = openFoam().readLagrangianVtk(path, file)
        dp = np.unique(data[:,1])
        for d in dp:
            psList = []
            index = np.where(np.isin(data[:,1], d))
            psInd = index[0]
            dd = data[psInd]
            psList.append(dd)
            for ps in psList:
                ddd = ps[0,1]
                aInd = np.where(np.greater_equal(ps[:,6], 0))
                pp = ps[aInd]
                print(len(pp))
                if len(pp) > 0:
                    x = pp[:,][:,6]
                    y = pp[:,][:,7]
                    z = pp[:,][:,8]
                    Vx = pp[:,][:,3]
                    Vy = pp[:,][:,4]
                    Vz = pp[:,][:,5]
                    
                    for i in range(len(Vz)):
                        if z[i] < -0.495:
                            print(Vz[i])
                            Vz[i] = 0
                    
                    mVx = np.mean(Vx)
                    mVy = np.mean(Vy)
                    mVz = np.mean(Vz)
                    
                    xm = np.mean(x)
                    ym = np.mean(y)
                    zm = np.mean(z)
                elif len(pp) == 0:
                    mVx = 0
                    mVy = 0
                    mVz = 0
                    
                    xm = 0
                    ym = 0
                    zm = 0
            
            indT = np.where(np.isin(allFileList, file))
            meanData.append([t[indT[0][0]], ddd,mVx, mVy, mVz, xm,ym,zm])
        print(file)
    resList.append(meanData)


for ps in range(len(resList[0])):
    l = []
    for t in resList:
        data = t[ps]
        l.append(data)
        n = np.asarray(l)
        s = int(np.round((n[0,1] * 10e5), 4))
        zero = np.copy(n[0,:])
        zero[0] = 0.0
        zero[2:8] = 0.0
        zero = np.reshape(zero, (1, -1))
        n = np.concatenate((zero, n))
    np.savetxt(fileName + str(s) + '.dat', n)

# a = openFoam().readLagrangianVtk(dirPath, allFileList[-1])
# index = np.where(np.isin(a[:,1], dp[-1]))
# psInd = index[0]
# aa = a[psInd]

# aInd = np.where(np.greater_equal(aa[:,6], 0))
# aaa = aa[aInd]
# plt.plot(aaa[:,6],aaa[:,8], ',')
# plt.xlim(0,1.6)
# plt.ylim(-0.5,0.5)
# plt.show()

