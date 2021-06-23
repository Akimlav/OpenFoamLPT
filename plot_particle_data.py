#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:24:13 2021

@author: akimlavrinenko
"""
import matplotlib.pyplot as plt
import numpy as np

dirPath1 = '/Users/akimlavrinenko/Documents/coding/data/cought/DNS_cough_data/'
dirPath2 = '/Users/akimlavrinenko/Documents/coding/data/cought/RANS_cough_data/'

psList = [4,8,16,32,64,128,256]

fig, axs = plt.subplots(7, figsize=(5, 15))
for i in range(7):
    dns = np.genfromtxt(dirPath1 + 'DNSPos' + str(psList[i]) + '.dat')
    rans = np.genfromtxt(dirPath2 + 'pData_' + str(psList[i]) + '.dat')
    print(len(dns), len(rans))
    axs[i].plot(dns[:, 3], dns[:, 2],'b--' ,label='DNS ' + str(psList[i]))
    axs[i].plot(rans[:,5], rans[:,7] ,'r-')
    axs[i].set_title('diameter ' + str(psList[i]) + ' \u03BC' + 'm')
    axs[i].grid(True)
fig.suptitle('mean cloud position', fontsize=14)
fig.tight_layout()
plt.show()

fig, axs = plt.subplots(7, figsize=(5, 15))
for i in range(7):
    dns = np.genfromtxt(dirPath1 + 'DNSVel.dat')
    rans = np.genfromtxt(dirPath2 + 'pData_' + str(psList[i]) + '.dat')
    axs[i].plot(dns[:, 0], dns[:, i+9],'b--' ,label='DNS ' + str(psList[i]))
    axs[i].plot(rans[:,0], rans[:,2] ,'r-')
    axs[i].set_title('diameter ' + str(psList[i]) + ' \u03BC' + 'm')
    axs[i].grid(True)
fig.suptitle('Vx', fontsize=14)
fig.tight_layout()
plt.show()


fig, axs = plt.subplots(7, figsize=(5, 15))
for i in range(7):
    dns = np.genfromtxt(dirPath1 + 'DNSVel.dat')
    rans = np.genfromtxt(dirPath2 + 'pData_' + str(psList[i]) + '.dat')
    axs[i].plot(dns[:, 0], dns[:, i+22],'b--' ,label='DNS ' + str(psList[i]))
    axs[i].plot(rans[:,0], rans[:,4] ,'r-')
    axs[i].set_title('diameter ' + str(psList[i]) + ' \u03BC' + 'm')
    axs[i].grid(True)
fig.suptitle('Vz', fontsize=14)
fig.tight_layout()
plt.show()