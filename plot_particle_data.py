#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:24:13 2021

@author: akimlavrinenko
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

dirPath1 = '/Users/akimlavrinenko/Documents/coding/data/cough/DNS_cough_data/'
dirPath2 = '/Users/akimlavrinenko/Documents/coding/data/cough/RANS_cough_data/'

cloud1 = np.genfromtxt(dirPath2 + '1kk_cloud.dat')
cloud1b = np.genfromtxt(dirPath2 + '1kkb_cloud.dat')
cloud10 = np.genfromtxt(dirPath2 + '10kk_cloud.dat')
cloud17 = np.genfromtxt(dirPath2 + '17kk_cloud.dat')
cloudDns = np.genfromtxt(dirPath1 + 'DNS_cloud.dat').T


# cloud
# dc1 = np.vstack((cloud1[:,1], cloud1[:,2])).T
# np.savetxt('rans1_cloud.dat', dc1)
# dc10 = np.vstack((cloud10[:,1], cloud10[:,2])).T
# np.savetxt('rans10_cloud.dat', dc10)
# dc17 = np.vstack((cloud17[:,1], cloud17[:,2])).T
# np.savetxt('rans17_cloud.dat', dc17)
# dc0 = np.vstack((cloudDns[:,0], cloudDns[:,1])).T
# np.savetxt('dns_cloud.dat', dc0)

psList = [4,8,16,32,64,128,256]
fontP = FontProperties()
fontP.set_size('xx-small')



# ([t, d,mVx, mVy, mVz, xm,ym,zm])
fig, axs = plt.subplots(7, figsize=(5, 15))
for i in range(7):

    dns = np.genfromtxt(dirPath1 + 'DNSPos' + str(psList[i]) + '.dat')
    rans1 = np.genfromtxt(dirPath2 + '1kk_pData' + str(psList[i]) + '.dat')
    rans1b = np.genfromtxt(dirPath2 + '1kk_bump_pData' + str(psList[i]) + '.dat')
    # rans1noDisp = np.genfromtxt(dirPath2 + '1kk_pData_noDisp' + str(psList[i]) + '.dat')
    # test = np.genfromtxt(dirPath2 + 'test_10kk_pData_' + str(psList[i]) + '.dat')
    # rans10 = np.genfromtxt(dirPath2 + '10kk_pData' + str(psList[i]) + '.dat')
    # rans17 = np.genfromtxt(dirPath2 + '17kk_pData' + str(psList[i]) + '.dat')
    # print(np.shape(rans17))
    # indDns = np.where(np.logical_and(dns[:,0] >= 1.648, dns[:,0] <= 1.652))
    # indRans1 = np.where(np.logical_and(rans1[:,0] >= 1.65, rans1[:,0] <= 1.655))
    # indRans10 = np.where(np.logical_and(rans10[:,0] >= 1.65, rans10[:,0] <= 1.655))
    # indRans17 = np.where(np.logical_and(rans17[:,0] >= 1.65, rans17[:,0] <= 1.655))
    # print(np.shape(dns), indDns[0])
    # if len(indDns[0]) > 0:
    #     dns = dns[:indDns[0][0]]
    # rans1 = rans1[:indRans1[0][0]]
    # rans10 = rans10[:indRans10[0][0]]
    # rans17 = rans17[:indRans17[0][0]]
    # else:
    # ddns = np.vstack((dns[:, 3],dns[:, 2])).T
    # np.savetxt('dns_' + str(psList[i]) + '.dat', ddns)
    # drans1 = np.vstack((rans1[:,5],rans1[:,7])).T
    # np.savetxt('rans1_' + str(psList[i]) + '.dat', drans1)
    # drans10 = np.vstack((rans10[:,5], rans10[:,7])).T
    # np.savetxt('rans10_' + str(psList[i]) + '.dat', drans10)
    # drans17 = np.vstack((rans17[:,5], rans17[:,7])).T
    # np.savetxt('rans17_' + str(psList[i]) + '.dat', drans17)

    
    # ddns = np.vstack((dns[201, 3], dns[201, 2])).T
    # np.savetxt('dot_dns_' + str(psList[i]) + '.dat', ddns)
    # drans1 = np.vstack((rans1[16,5], rans1[16,7])).T
    # np.savetxt('dot_rans1_' + str(psList[i]) + '.dat', drans1)
    # drans10 = np.vstack((rans10[16,5], rans10[16,7])).T
    # np.savetxt('dot_rans10_' + str(psList[i]) + '.dat', drans10)
    # drans17 = np.vstack((rans17[8,5], rans17[8,7])).T
    # np.savetxt('dot_rans17_' + str(psList[i]) + '.dat', drans17)
    
    axs[i].plot(dns[:, 3], dns[:, 2],'black' ,label='DNS ' + str(psList[i]))
    axs[i].plot(dns[201, 3], dns[201, 2],color = 'black',marker = 'o')
    
    axs[i].plot(cloudDns[:,0], cloudDns[:,1] ,'k--')
    axs[i].plot(cloud1[:,1], cloud1[:,2] ,'b--')
    axs[i].plot(cloud1b[:,0], cloud1b[:,1] ,'c--')
    # axs[i].plot(cloud1b[:,1], cloud1b[:,2] ,'b--')
    # axs[i].plot(cloud10[:,1], cloud10[:,2] ,'r--')
    # axs[i].plot(cloud17[:,1], cloud17[:,2] ,'g--')
    axs[i].plot(rans1[:,5], rans1[:,7] ,'b')
    axs[i].plot(rans1b[:,5], rans1b[:,7] ,'c')
    axs[i].plot(rans1[16,5], rans1[16,7] ,'bo')
    axs[i].plot(rans1b[16,5], rans1b[16,7] ,'co')
    
    # axs[i].plot(rans1noDisp[:,5], rans1noDisp[:,7] ,'m')
    # axs[i].plot(rans1[16,5], rans1[16,7] ,'bo')
    
    # axs[i].plot(rans10[:,5], rans10[:,7] ,'r')
    # axs[i].plot(rans10[16,5], rans10[16,7] ,'ro')
    # axs[i].plot(rans17[:,5], rans17[:,7] ,'g')
    # axs[i].plot(rans17[8,5], rans17[8,7] ,'go')
    axs[i].set_xlim(0,0.85)
    axs[i].set_xlabel(' x [m]')
    axs[i].set_ylabel('z [m]')
    axs[i].set_title('diameter ' + str(psList[i]) + ' \u03BC' + 'm')
    axs[i].grid(True)
fig.suptitle('mean cloud position', fontsize=14)
# fig.legend(title = 'cloud', bbox_to_anchor=(0.75, 0.88), loc='upper left', prop=fontP)
fig.tight_layout()
plt.savefig('XZ_distance.png', dpi=150)
plt.show()

fig, axs = plt.subplots(7, figsize=(5, 15))
for i in range(7):
    dns = np.genfromtxt(dirPath1 + 'DNSVel.dat')
    rans1 = np.genfromtxt(dirPath2 + '1kk_pData' + str(psList[i]) + '.dat')
    rans1b = np.genfromtxt(dirPath2 + '1kk_bump_pData' + str(psList[i]) + '.dat')
    rans1noDisp = np.genfromtxt(dirPath2 + '1kk_pData_noDisp' + str(psList[i]) + '.dat')
    rans10 = np.genfromtxt(dirPath2 + '10kk_pData' + str(psList[i]) + '.dat')
    rans17 = np.genfromtxt(dirPath2 + '17kk_pData' + str(psList[i]) + '.dat')
    axs[i].plot(dns[:, 0], dns[:, i+9],'k-' ,label='DNS ' + str(psList[i]))
    axs[i].plot(rans1[:,0], rans1[:,2] ,'b-')
    axs[i].plot(rans1b[:,0], rans1b[:,2] ,'c-')
    # axs[i].plot(rans1noDisp[:,0], rans1noDisp[:,2] ,'m-')
    # axs[i].plot(rans10[:,0], rans10[:,2] ,'r-')
    # axs[i].plot(rans17[:,0], rans17[:,2] ,'g-')
    axs[i].set_title('diameter ' + str(psList[i]) + ' \u03BC' + 'm')
    axs[i].grid(True)
    axs[i].set_xlabel(' t [s]')
    axs[i].set_ylabel('Vx [m/s]')
    axs[i].set_xlim(-0.05,1.6)
    # s = np.vstack((dns[:, 0], dns[:, i+9])).T
    # np.savetxt('dns_Vx_'+ str(psList[i]) + '.dat', s)
    # ss = np.vstack((rans1[:,0], rans1[:,2])).T
    # np.savetxt('rans1_Vx_'+ str(psList[i]) + '.dat', ss)
    # sss = np.vstack((rans10[:,0], rans10[:,2])).T
    # np.savetxt('rans10_Vx_'+ str(psList[i]) + '.dat', sss)
    # ssss = np.vstack((rans17[:,0], rans17[:,2])).T
    # z = [0.025, 0]
    # ssss = np.insert(ssss,1, z, axis = 0)
    # np.savetxt('rans17_Vx_'+ str(psList[i]) + '.dat', ssss)
fig.suptitle('Vx', fontsize=14)
fig.tight_layout()
plt.savefig('Vx.png', dpi=150)
plt.show()


fig, axs = plt.subplots(7, figsize=(5, 15))
for i in range(7):
    dns = np.genfromtxt(dirPath1 + 'DNSVel.dat')
    rans1 = np.genfromtxt(dirPath2 + '1kk_pData' + str(psList[i]) + '.dat')
    rans1b = np.genfromtxt(dirPath2 + '1kk_bump_pData' + str(psList[i]) + '.dat')
    rans1noDisp = np.genfromtxt(dirPath2 + '1kk_pData_noDisp' + str(psList[i]) + '.dat')
    rans10 = np.genfromtxt(dirPath2 + '10kk_pData' + str(psList[i]) + '.dat')
    rans17 = np.genfromtxt(dirPath2 + '17kk_pData' + str(psList[i]) + '.dat')
    axs[i].plot(dns[:, 0], dns[:, i+22],'k-' ,label='DNS ' + str(psList[i]))
    axs[i].plot(rans1[:,0], rans1[:,4] ,'b-')
    axs[i].plot(rans1b[:,0], rans1b[:,4] ,'c-')
    # axs[i].plot(rans1noDisp[:,0], rans1noDisp[:,4] ,'m-')
    # axs[i].plot(rans10[:,0], rans10[:,4] ,'r-')
    # axs[i].plot(rans17[:,0], rans17[:,4] ,'g-')
    axs[i].set_title('diameter ' + str(psList[i]) + ' \u03BC' + 'm')
    axs[i].grid(True)
    # k = np.vstack((dns[:, 0], dns[:, i+22])).T
    # np.savetxt('dns_Vz_'+ str(psList[i]) + '.dat', k)
    # kk = np.vstack((rans1[:,0], rans1[:,4])).T
    # np.savetxt('rans1_Vz_'+ str(psList[i]) + '.dat', kk)
    # kkk = np.vstack((rans10[:,0], rans10[:,4])).T
    # np.savetxt('rans10_Vz_'+ str(psList[i]) + '.dat', kkk)
    # kkkk = np.vstack((rans17[:,0], rans17[:,4])).T
    # zz = [0.025, 0]
    # kkkk = np.insert(kkkk,1, z, axis = 0)
    # np.savetxt('rans17_Vz_'+ str(psList[i]) + '.dat', kkkk)
fig.suptitle('Vz', fontsize=14)
fig.tight_layout()
plt.savefig('Vz.png', dpi=150)
plt.show()

# fig, axs = plt.subplots(7, figsize=(5, 15))
# for i in range(7):
#     dns = np.genfromtxt(dirPath1 + 'DNSVel.dat')
#     rans1 = np.genfromtxt(dirPath2 + '1kk_pData_' + str(psList[i]) + '.dat')
#     rans10 = np.genfromtxt(dirPath2 + '10kk_pData_' + str(psList[i]) + '.dat')
#     # axs[i].plot(dns[:, 0], dns[:, i+22],'b--' ,label='DNS ' + str(psList[i]))
#     axs[i].plot(rans1[:,0], rans1[:,3] ,'r-')
#     axs[i].plot(rans10[:,0], rans10[:,3] ,'r--')
#     axs[i].set_title('diameter ' + str(psList[i]) + ' \u03BC' + 'm')
#     axs[i].grid(True)
# fig.suptitle('Vy', fontsize=14)
# fig.tight_layout()
# plt.savefig('Vy.png', dpi=150)
# plt.show()