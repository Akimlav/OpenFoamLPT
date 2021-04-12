#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:04:26 2021

@author: akimlavrinenko
"""
import numpy as np
import matplotlib.pyplot as plt
from NEK5000_func_lib import fast_scandir, listfile, find_in_list_of_list

dirpath = '/Users/akimlavrinenko/Documents/coding/data/OF_ps_data/test0_k0/fields_k0'
fold_name = 'reactingCloud1'
folders = fast_scandir(dirpath)
folders = [word for word in folders if fold_name in word]
folders.sort()
folders = folders[::2]
listOfFileList, allFileList = listfile(folders, 'U')



def renameU(timeScale):
    folders = fast_scandir(dirpath)
    folders = [word for word in folders if fold_name in word]
    folders.sort()
    folders = folders[::2]
    listOfFileList, allFileList = listfile(folders, 'U')
    for i in range(1,len(allFileList)):
        f1 = open(folders[i-1] + '/' + 'U', 'r')
        f2 = open(folders[i-1] + '/' + 'U_' + str(np.round(i * timeScale, 3)) + '.dat', 'w')
        
        for line in f1:
            f2.write(line.replace('(', '').replace(')', ''))   
        f1.close()
        f2.close()

folders = fast_scandir(dirpath)
folders = [word for word in folders if fold_name in word]
folders.sort()
folders = folders[::2]
listOfFileList, allFileList = listfile(folders, '.dat')

# renameU(0.05)


mVxList = []
for file in allFileList:
    ind = find_in_list_of_list(listOfFileList, file)
    if file in listOfFileList[ind[0]]:
        path = folders[ind[0]] + '/'
        data=np.genfromtxt(path + file,skip_header=20, skip_footer=1, invalid_raise = False)
        Vx = data[:,][:,2]
        mVx = np.mean(Vx)
        mVxList.append(mVx)


t = np.linspace(0.05,(len(allFileList)*0.05),num = len(allFileList))
t = np.append(0, t)
mVxList.insert(0,0)
#DNS data
velocityOrig=np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/DNS_cough/MeanVel.txt', invalid_raise = False)
velocity4 = velocityOrig[:,22] * 4.8
t4 = velocityOrig[:,0] * 0.0002
#end of DNS data



fig, ax = plt.subplots(figsize=(5, 5))
ax.grid(True)
plt.plot(t, mVxList,'o-',label="RANS 4e-6 m")
plt.plot(t4, velocity4,'-',label="DNS 4e-6 m")
plt.xlabel('Time, sec')
plt.ylabel('Vx, m/s')
# plt.ylim(0,3.0)
plt.xlim(0,1.7)
plt.legend(loc="upper right")
plt.savefig('DNS_RANS_mean_velocity.png',dpi= 150)
plt.show()
