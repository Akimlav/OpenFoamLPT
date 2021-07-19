#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:32:04 2021

@author: akimlavrinenko
"""


import matplotlib.pyplot as plt
import numpy as np
# velocityOrig=np.genfromtxt('/Users/akimlavrinenko/Documents/coding/data/MeanVel.txt', invalid_raise = False)
positionOrig=np.genfromtxt('//Users/akimlavrinenko/Documents/coding/data/cough/DNS_cough_orig_data/DNS_centroid.csv', invalid_raise = False)

dnsX = positionOrig[:,8]*0.02
dnsY = positionOrig[:,7]*0.02
save = [dnsX,dnsY]
np.savetxt('dns_cloud.dat', save)
t = positionOrig[:,0]

# positionX4 = positionOrig[:,3] * 0.002
# positionY4 = positionOrig[:,2] * 0.002
# velocity4 = velocityOrig[:,8] * 4.8
# t4 = velocityOrig[:,0] * 0.0002


# fig, ax = plt.subplots(figsize=(5, 5))
# ax.grid(True)
# plt.plot(positionX4, positionY4,'-',label="DNS 4e-6 m")
# plt.xlabel('Time, sec')
# plt.ylabel('Vx, m/s')
# plt.ylim(0,2.8)
# plt.xlim(0,1.7)
# plt.legend(loc="upper right")
# plt.savefig('ps_mean_velocity.png',dpi= 150)
# plt.show()