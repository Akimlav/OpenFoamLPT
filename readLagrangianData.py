#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 19:50:25 2021

@author: akimlavrinenko
"""
from openFoamClass import openFoam
import numpy as np

dirPath = '/Users/akimlavrinenko/Documents/coding/data/cought/lagrangian_data/VTK/lagrangian/'

foldName = 'reactingCloud1'

folders = openFoam().fast_scandir(dirPath)
folders = [word for word in folders if foldName in word]
folders.sort()
listOfFileList, allFileList = openFoam().listfile(folders, '.vtk')
allFileList.sort()

timeStep = 0.1
t = np.linspace(1*timeStep ,((len(allFileList))*timeStep),num = len(allFileList))

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
                
                dd = ps[0,1]
                x = ps[:,][:,6]
                y = ps[:,][:,7]
                z = ps[:,][:,8]
                Vx = ps[:,][:,3]
                Vy = ps[:,][:,4]
                Vz = ps[:,][:,5]
                
                mVx = np.mean(Vx)
                mVy = np.mean(Vy)
                mVz = np.mean(Vz)
                
                xm = np.mean(x)
                ym = np.mean(y)
                zm = np.mean(z)
                indT = np.where(np.isin(allFileList, file))
                meanData.append([t[indT[0][0]], dd,mVx, mVy, mVz, xm,ym,zm])
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
    np.savetxt('pData_' + str(s) + '.dat', n)

