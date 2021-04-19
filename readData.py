#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:24:13 2021

@author: akimlavrinenko
"""
import matplotlib.pyplot as plt
import numpy as np


#read DNS data
DNSVelOrig=np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/DNS_cough/MeanVel.txt', invalid_raise = False)
DNSPosOrig4um=np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/DNS_cough/s1e1_xc.dat', invalid_raise = False)
# DNSPosOrig8um=np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/DNS_cough/s2e1_xc.dat', invalid_raise = False)
# DNSPosOrig16um=np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/DNS_cough/s3e1_xc.dat', invalid_raise = False)
DNSPosOrig32um=np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/DNS_cough/s4e1_xc.dat', invalid_raise = False)


#read URANSE data
RANSPos4um = np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/OF_ps_data/pure_data/pos_4um.dat', invalid_raise = False)
RANSPos4umBKEps = np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/OF_ps_data/pure_data/pos_4um_bKEps.dat', invalid_raise = False)
RANSPos32um = np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/OF_ps_data/pure_data/pos_32um.dat', invalid_raise = False)
RANSVel4um = np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/OF_ps_data/pure_data/vel_4um.dat', invalid_raise = False)
RANSVel4umBKEps = np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/OF_ps_data/pure_data/vel_4um_bKEps.dat', invalid_raise = False)
RANSVel32um = np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/OF_ps_data/pure_data/vel_32um.dat', invalid_raise = False)


#DNS data tramsform

temp = DNSPosOrig4um[:, 0].copy()
DNSPos4um = DNSPosOrig4um * 0.02 
DNSPos4um[:, 0] = temp * 0.002

temp = DNSPosOrig32um[:, 0].copy()
DNSPos32um = DNSPosOrig32um * 0.02 
DNSPos32um[:, 0] = temp * 0.0002

temp = DNSVelOrig[:, 0].copy()
DNSVel = DNSVelOrig * 4.8
DNSVel[:, 0] = temp * 0.0002


# DNSPos4um = DNSPos4um[:,]* 0.02
#end of DNS data

t1 = np.linspace(0.05,(len(RANSPos4um)*0.05),num = len(RANSPos4um))
t2 = np.linspace(0.0,(len(RANSVel4um)*0.05),num = len(RANSVel4um)+1)
t2 = t2[:-1]
t3 = np.linspace(0.0,(len(RANSVel32um)*0.025),num = len(RANSVel32um)+1)
t3 = t3[:-1]
t4 = np.linspace(0.0,(len(RANSVel4umBKEps)*0.025),num = len(RANSVel4umBKEps)+1)
t4 = t4[:-1]

fig, axs = plt.subplots(3, figsize=(5, 10))
axs[0].plot(DNSPos4um[:, 3], DNSPos4um[:, 2], label="DNS 4")
axs[0].plot(DNSPos32um[:, 3], DNSPos32um[:, 2], '-', label='DNS 32')
axs[0].plot(RANSPos4um[:, 0], RANSPos4um[:, 2], '--',label="RANS 4")
axs[0].plot(RANSPos4umBKEps[:, 0], RANSPos4umBKEps[:, 2], '--',label="RANS bKEps 4")
axs[0].plot(RANSPos32um[:, 0], RANSPos32um[:, 2], '--',label="RANS 32")
axs[0].set_title('cloud mean position')
axs[0].grid(True)
# axs[0].legend(bbox_to_anchor=(1.05, 0.95))
axs[1].plot(DNSVel[:, 0], DNSVel[:, 8], '-',label="DNS 4")
axs[1].plot(DNSVel[:, 0], DNSVel[:, 11], '-',label="DNS 32")
axs[1].plot(t2, RANSVel4um[:,0], '--',label="RANS 4")
axs[1].plot(t4, RANSVel4umBKEps[:,0], '--',label="RANS 4 bKEps")
axs[1].plot(t3, RANSVel32um[:,0], '--',label="RANS 32")
axs[1].legend(bbox_to_anchor=(0.95, 0.95))
axs[1].set_title('mean Vx')
axs[1].set_xlim([0, 1.6])
axs[1].grid(True)
axs[2].set_title('mean Vz')
axs[2].plot(DNSVel[:, 0], DNSVel[:, 22], '-',label="DNS 4")
axs[2].plot(DNSVel[:, 0], DNSVel[:, 25], '-',label="DNS 4")
axs[2].plot(t2, RANSVel4um[:,2], '--',label="RANS 4")
axs[2].plot(t4, RANSVel4umBKEps[:,2], '--',label="RANS 4 bKEps")
axs[2].plot(t3, RANSVel32um[:,2], '--',label="RANS 32")
axs[2].set_xlim([0, 1.6])
axs[2].grid(True)
plt.savefig('DNS_RANS.png', dpi= 200)
plt.show()
