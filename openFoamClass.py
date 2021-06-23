#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 10:08:02 2021

@author: akimlavrinenko
"""
import numpy as np
import vtk
from os import listdir, scandir
from vtk.util.numpy_support import vtk_to_numpy

class openFoam:
    
    def fast_scandir(self, dirname):
        subfolders= [f.path for f in scandir(dirname) if f.is_dir()]
        for dirname in list(subfolders):
            subfolders.extend(openFoam().fast_scandir(dirname))
        return subfolders

    def listfile(self, folders, file):
        allFileList = []
        listOfFileList = []
        for folder in folders:
            fileList = [name for name in listdir(folder) if name.endswith(file)]
            fileList.sort()
            listOfFileList.append(fileList)
            allFileList = allFileList + fileList
        return (listOfFileList, allFileList)

    def find_in_list_of_list(self, mylist, char):
        for sub_list in mylist:
            if char in sub_list:
                return (mylist.index(sub_list), sub_list.index(char))
        raise ValueError("'{char}' is not in list".format(char = char))
    
    #renames U files to U_timeStep.extension
    def renameLPTData(self, dirPath, foldName, timeScale, extension):
        folders = openFoam().fast_scandir(dirPath)
        folders = [word for word in folders if foldName in word]
        folders.sort()
        folders = folders[::2]
        listOfFileList, allFileList = openFoam().listfile(folders, 'U')
        for i in range(1,len(allFileList)+1):
            f1 = open(folders[i-1] + '/' + 'U', 'r')
            f2 = open(folders[i-1] + '/' + 'U_' + str(np.round(i * timeScale, 3)) + extension, 'w')
            
            for line in f1:
                f2.write(line.replace('(', '').replace(')', ''))   
            f1.close()
            f2.close()
    #returns 2d array with Time, x, y, z velocities
    def readLPTVelocity(self, dirPath, foldName, timeScale, extension):
            folders = openFoam().fast_scandir(dirPath)
            folders = [word for word in folders if foldName in word]
            folders.sort()
            folders = folders[::2]
            listOfFileList, allFileList = openFoam().listfile(folders, extension)
            meanDataList = []
            for file in allFileList:
                ind = openFoam().find_in_list_of_list(listOfFileList, file)
                if file in listOfFileList[ind[0]]:
                    path = folders[ind[0]] + '/'
                    data=np.genfromtxt(path + file,skip_header=20, skip_footer=1, invalid_raise = False)
                    Vx = data[:,][:,0]
                    Vy = data[:,][:,1]
                    Vz = data[:,][:,2]
                    mVx = np.mean(Vx)
                    mVy = np.mean(Vy)
                    mVz = np.mean(Vz)
                    meanDataList.append([mVx, mVy, mVz])
            
            meanDataList.insert(0,[0,0,0])
            meanDataList = np.asarray(meanDataList)
            
            t = np.linspace(timeScale ,(len(allFileList)*timeScale),num = len(allFileList))
            t = np.append(0, t)
            t = np.reshape(t, (-1, 1))
            
            data = np.concatenate((t,meanDataList), axis = 1)
            return data

    def readLPTPositions(self, dirPath, foldName, timeScale):
        
        folders = openFoam().fast_scandir(dirPath)
        folders = [word for word in folders if foldName in word]
        folders.sort()
        listOfFileList, allFileList = openFoam().listfile(folders, '.vtk')
        allFileList.sort()
    
        meanData = []
    
        for file in allFileList:
            ind = openFoam().find_in_list_of_list(listOfFileList, file)
            if file in listOfFileList[ind[0]]:
                path = folders[ind[0]] + '/'
                data = openFoam().vtkReader(path + file)
                coords = np.asarray(data)
                # print(len(coords), file)
                x = coords[:,][:,0]
                y = coords[:,][:,1]
                z = coords[:,][:,2]
                xm = np.mean(x)
                ym = np.mean(y)
                zm = np.mean(z)
                meanData.append([xm,ym,zm])
        
        meanData = np.asarray(meanData)
        meanData[0,:] = [0,0,0]
        
        t = np.linspace(0 ,((len(allFileList)-1)*timeScale),num = len(allFileList))
        t = np.reshape(t, (-1, 1))
        data = np.concatenate((t,meanData), axis = 1)
        return data

    def readLagrangianVtk(self, dirPath, file):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(dirPath + file)
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllTensorsOn()
        reader.Update()
        
        origId = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(1))
        d = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(4))
        T = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(14))
        points = np.array(reader.GetOutput().GetPoints().GetData())
        U = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray('U'))
        
        Up = np.concatenate((U, points), axis = 1)
        idDT = np.asarray([origId, d, T]).T
        data = np.concatenate((idDT, Up), axis = 1)
        return data
        
        