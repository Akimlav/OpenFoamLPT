#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:34:30 2021

@author: akimlavrinenko
"""

#NEK 5000 functions library
#!/usr/bin/python3.8

from sys import argv
from os import system,remove,cpu_count, listdir, scandir

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gzip
import shutil
from multiprocessing import Pool
import random
import glob
import os
import sys
from time import time
from scipy.stats import norm
import statistics as st
import multiprocessing 
import math as m
import itertools

# print('enter path to folder with files:')
# path = input() + '/'

def slice_per(source, step):
    return [source[i::step] for i in range(step)]

def readParticleFile(pfilename):
  # print("Reading "+pfilename)
  isgzip = 0
  if pfilename[-2:]=='gz':
    pfile = gzip.open(pfilename,"rb")
    kk = open(pfilename[:-3],'wb')
    shutil.copyfileobj(pfile,kk)
    pfile.close()
    kk.close()
  
    pfile = open(pfilename[:-3],'rb')
    isgzip = 1
  else:
    pfile = open(pfilename,'rb')
  #
  
  # The first entry in the fortran binary file is a 4 byte word, apparently.
  # I don't know why.
  dum = np.fromfile(pfile,dtype=np.int32,count=1)
  
  time = np.fromfile(pfile,dtype=np.float64,count=1)[0]
  counters = np.fromfile(pfile,dtype=np.int32,count=2)
  
  # Number of particles seeded for each set of data
  nseedParticles = counters[0]
  # Number of particle sizes
  nsizes = counters[1]
  
  # The first entry in the fortran binary file is a 4 byte word, apparently.
  # I don't know why.
  dum = np.fromfile(pfile,dtype=np.int32,count=3)
  counters = np.fromfile(pfile,dtype=np.int32,count=6)
  
  # Number of fields per particle
  nfields = counters[3]
  # Total number of particles
  nparticles = counters[4]
  print('Reading: ' + pfilename)
  print("Time: "+str(time*0.02/4.8)+" secs")
  print("-------------------------------------------------------------------")
  # This is the data structure for each particle
  dataType = np.dtype([('dum'  ,np.int64    ), # Dummy 8 byte word, unknown origin
                       ('batch',np.float64  ), # Particle batch number
                       ('sp'   ,np.float64,(1)), # Particle size
                       ('xp'   ,np.float64,(3)), # Particle coordinates
                       ('up'   ,np.float64,(3)), # Particle speed
                       ('he'   ,np.float64  ), # 
                       ('hp'   ,np.float64  ), #
                       ('rest' ,np.float64,nfields-10)]) # Fields not used at the moment
  
  # Read the data for all particles
  pdata = np.fromfile(pfile,dtype=dataType,count=nparticles)

  pfile.close()
  
  # remove the gunzipped file
  if isgzip==1: 
    remove(pfilename[:-3])
  
  # Now discard the empty part of pdata
  nfull = nparticles//2
  ndelta = nparticles//4
  while True:
    if nfull==nparticles-1 or (pdata['batch'][nfull]!=0. and pdata['batch'][nfull+1]==0.):
      break
    else:
      if pdata['batch'][nfull]!=0.:
        nfull += ndelta
      else:
        nfull -= ndelta
      if ndelta>1: ndelta = ndelta//2
    #endif
  #end while
  
  pdata = np.resize(pdata,(nfull+1,))
  
  nbatch = int(pdata['batch'][-1])
  
  # Reshape the array with nbatches, with nsizes of nseedParticles each
  pdata = pdata.reshape(nbatch,nsizes,nseedParticles)
  
  return time,pdata


def saveData(pfilename):
    time,pdata = readParticleFile(pfilename)
    print(time)
    # np.savetxt('data.dat', pdata)
    


# end def readParticleFile

def plotParticle(pfilename,time,pdata,particleSize):
  fig = plt.figure()
  
  ax = fig.add_subplot(111,projection='3d')
  
  xp = []
  yp = []
  zp = []
  
  for ibatch in range(0,pdata.shape[0]):
    for ipart in range(0,pdata.shape[2]):
      xpart = pdata[ibatch,particleSize,ipart]['xp']
      xp.append(xpart[0])
      yp.append(xpart[1])
      zp.append(xpart[2])
  
  ax.scatter(xp,yp,zp,marker='.',s=1,color='r')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  
  ax.set_xlim(-0.5,0.5)
  ax.set_ylim(-0.5,0.5)
  ax.set_zlim(-0.5,0.5)
      
  # Set a desired view angle 
  ax.view_init(90,270)
  plt.savefig(pfilename[:-3]+'.png')
  plt.close()
#end def plotParticle

def readAndPlot(pfilename):
  time,pdata = readParticleFile(pfilename)
  print('Time is: %.3f '  % (time))
  particleSize = 0
    
  plotParticle(pfilename,time,pdata,particleSize)
  (pdata)

###################################################################################################################

# fileList = [name for name in os.listdir(path) if name.endswith(".3D")]
# fileList.sort()

def particleCoords (path, fileName, particleSize):
    time,pdata = readParticleFile(str(path) + str(fileName))
    
    xyzp = []
            
    for ibatch in range(0,pdata.shape[0]):
        for ipart in range(0,pdata.shape[2]):
          xpart = pdata[ibatch,particleSize,ipart]['xp']
          xyzp.append(xpart)
    p_coords = np.asarray(xyzp)
    return(time, p_coords)

def particleCoordsNew (path, fileName):
    time,pdata = readParticleFile(str(path) + str(fileName))
    
    xyzp = []
            
    for ibatch in range(0,pdata.shape[0]):
        for ipart in range(0,pdata.shape[2]):
            for ps in range(0,pdata.shape[1]): 
            
              xpart = pdata[ibatch,ps,ipart]['xp']
              xyzp.append(xpart)
        
        allp = [xyzp[z:z+pdata.shape[2]] for z in range(0, len(xyzp), pdata.shape[2])]
    
    return(time, allp)

def plotVideo (choose, n, Dimension, particleSize, center, radius, plotsmbl):
    start_time = time()
    xyzz = particleCoords(path, fileList[0], particleSize)
    index = np.random.choice(xyzz.shape[0], n, replace=False)
    for i in fileList:
        if choose.lower() in ['r', 'random', 'rnd']:
            xyz = particleCoords(path, i, particleSize)
            xyz1np = xyz[index]
        elif choose.lower() in ['all', 'a']:
                xyz = particleCoords(path, i, particleSize)
                xyz1np = np.asarray(xyz)
        elif choose.lower() in ['index', 'i']:
                xyz = particleCoords(path, i, particleSize)
                index = n
                xyz1np = xyz[index]
        elif choose.lower() in ['sphere', 's']:
            filtered = []
            for j in range(len(xyzz)):
                r1 = ((xyzz[j,0] - center[0])**2 + (xyzz[j,1] - center[1])**2 + (xyzz[j,2] - center[2])**2)**0.5
                if r1 <= radius:
                    filtered.append(xyzz[j,:])
            filtered = np.asarray(filtered)
            index = np.where(np.isin(xyzz[:,1], filtered))

            xyz = particleCoords(path, i, particleSize)
            xyz1np = xyz[index]
            print(index[:10])


        if Dimension.lower() in ['2d', '2D']:
            if xyz1np.shape[0] == 3:
                plt.plot(xyz1np[0],xyz1np[1], plotsmbl, markersize=0.5)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.xlim(-0.5,0.5)
                plt.ylim(-0.5,0.5)
                # plt.grid()
                plt.savefig(i[:-3] + '_2D.png', dpi=300)
                plt.clf()
            else:
                plt.plot(xyz1np[:,0],xyz1np[:,1], plotsmbl, markersize=0.5)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.xlim(-0.5,0.5)
                plt.ylim(-0.5,0.5)
                # plt.grid()
                plt.savefig(i[:-3] + '_2D.png', dpi=300)
                plt.clf()
                
        elif Dimension.lower() in ['3d', '3D']:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if xyz1np.shape[0] == 3:
                ax.plot(xyz1np[0],xyz1np[1],xyz1np[2], plotsmbl, markersize=0.5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-0.5,0.5)
                ax.set_ylim(-0.5,0.5)
                ax.set_zlim(-0.5,0.5)
                plt.savefig(i[:-3] + '_3D.png', dpi=300)
                plt.clf()
            else:
                ax.plot(xyz1np[:,0], xyz1np[:,1], xyz1np[:,2], plotsmbl, markersize=0.5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-0.5,0.5)
                ax.set_ylim(-0.5,0.5)
                ax.set_zlim(-0.5,0.5)
                plt.savefig(i[:-3] + '_3D.png', dpi=300)
                plt.clf()

    print('It was: %.3f seconds'  % (time() - start_time))
    np.savetxt('s_index.dat', index)

def plotVideoContiniue (Dimension,particleSize, plotsmbl):
    index1 = np.genfromtxt('s_index.dat', skip_header = 0, invalid_raise = False)
    index1 = index1.astype(int)
    start_time = time()
    print(index1[:10])
    for i in fileList:
        xyz = particleCoords(path, i, particleSize)
        xyz1np = xyz[index1]
        if Dimension.lower() in ['2d', '2D']:
            if xyz1np.shape[0] == 3:
                plt.plot(xyz1np[0],xyz1np[1], plotsmbl, markersize=0.5)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.xlim(-0.5,0.5)
                plt.ylim(-0.5,0.5)
                # plt.grid()
                plt.savefig(i[:-3] + '_2D.png', dpi=300)
                plt.clf()
            else:
                plt.plot(xyz1np[:,0],xyz1np[:,1], plotsmbl, markersize=0.5)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.xlim(-0.5,0.5)
                plt.ylim(-0.5,0.5)
                # plt.grid()
                plt.savefig(i[:-3] + '_2D.png', dpi=300)
                plt.clf()
                
        elif Dimension.lower() in ['3d', '3D']:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if xyz1np.shape[0] == 3:
                ax.plot(xyz1np[0],xyz1np[1],xyz1np[2], plotsmbl, markersize=0.5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-0.5,0.5)
                ax.set_ylim(-0.5,0.5)
                ax.set_zlim(-0.5,0.5)
                plt.savefig(i[:-3] + '_3D.png', dpi=300)
                plt.clf()
            else:
                ax.plot(xyz1np[:,0], xyz1np[:,1], xyz1np[:,2], plotsmbl, markersize=0.5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-0.5,0.5)
                ax.set_ylim(-0.5,0.5)
                ax.set_zlim(-0.5,0.5)
                plt.savefig(i[:-3] + '_3D.png', dpi=200)
                plt.clf()
    print('It was: %.3f seconds'  % (time() - start_time))


def plotTrajectory (choose, n, Dimension, particleSize, center, radius, plotsmbl):
    start_time = time()
    xyz1 = []
    xyzz = particleCoords(path, fileList[0], particleSize)
    index = np.random.choice(xyzz.shape[0], n, replace=False)
    if choose.lower() in ['r', 'random', 'rnd']:
        for i in fileList:
            xyz = particleCoords(path, i, particleSize)
            xyz_rnd = xyz[index]
            xyz1.append(xyz_rnd)
            xyz1np = np.asarray(xyz1)
    elif choose.lower() in ['all', 'a']:
        for i in fileList:
            xyz = particleCoords(path, i, particleSize)
            xyz1.append(xyz)
            xyz1np = np.asarray(xyz1)
            
    elif choose.lower() in ['index', 'i']:
        for i in fileList:
            xyz = particleCoords(path, i, particleSize)
            index = n
            xyz_rnd = xyz[index]
            xyz1.append(xyz_rnd)
            xyz1np = np.asarray(xyz1)
    elif choose.lower() in ['sphere', 's']:
        filtered = []
        for i in range(len(xyzz)):
           r1 = ((xyzz[i,0] - center[0])**2 + (xyzz[i,1] - center[1])**2 + (xyzz[i,2] - center[2])**2)**.5
           if r1 <= radius:
               filtered.append(xyzz[i,:])
        filtered = np.asarray(filtered)   
        index = np.where(np.isin(xyzz[:,1], filtered))  
        for i in fileList:
            xyz = particleCoords(path, i, particleSize)
            xyz_ind = xyz[index]
            xyz1.append(xyz_ind)
            xyz1np = np.asarray(xyz1)

    if Dimension.lower() in ['3d', '3D']:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(np.shape(xyz1np)[1]):
            if xyzz.shape[1] == 3:
                ax.plot(xyz1np[:,0],xyz1np[:,1],xyz1np[:,2], plotsmbl)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-0.5,0.5)
                ax.set_ylim(-0.5,0.5)
                ax.set_zlim(-0.5,0.5)
                
            else:
                a1 = xyz1np[:,i,:]
                ax.plot(a1[:,0],a1[:,1], a1[:,2], plotsmbl)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-0.5,0.5)
                ax.set_ylim(-0.5,0.5)
                ax.set_zlim(-0.5,0.5)
            ax.figure.savefig(str(choose) + '_' + str(n) + '_' + str(Dimension) + '_' +  str(particleSize) + '_' + str(center)  + '_' + str(radius) + '.png')
            # plt.show()
        
    elif Dimension.lower() in ['2d', '2D']:
        if xyz1np.shape[1] == 3:
            for i in range(np.shape(xyz1np)[1]):
                plt.plot(xyz1np[:,0],xyz1np[:,1], plotsmbl, linewidth=0.5)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.xlim(-0.5,0.5)
                plt.ylim(-0.5,0.5)
            # plt.grid()
            plt.savefig(str(choose) + '_' + str(n) + '_' + str(Dimension) + '_' +  str(particleSize) + '_' + str(center)  + '_' + str(radius) + '.png', dpi = 300)
            # plt.show()
        else:
            for i in range(np.shape(xyz1np)[1]):
                a1 = xyz1np[:,i,:]
                plt.plot(a1[:,0],a1[:,1], plotsmbl, linewidth=0.5)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.xlim(-0.5,0.5)
                plt.ylim(-0.5,0.5)
            # plt.grid()
            plt.savefig(str(choose) + '_' + str(n) + '_' + str(Dimension) + '_' +  str(particleSize) + '_' + str(center)  + '_' + str(radius) + '.png', dpi = 300)
            # plt.show()
    
            
    print('It was: %.3f seconds'  % (time() - start_time))
    return(xyz1np)

def totalDist(filename, extension, particleSize):
    total_dist = 0
    for i in range(1, len(fileList)-1):
        if i  < len(fileList):
            time0,pdata0 = readParticleFile(path + filename + "{number:05}".format(number=i + int(fileList[0][-8:-3])) + '.' + extension)
            time1,pdata1 = readParticleFile(path + filename + "{number:05}".format(number=i+1 +int(fileList[0][-8:-3])) + '.' + extension)
            for ibatch in range(0,pdata0.shape[0]):
                x0 = pdata0['xp'][ibatch][particleSize][:,0]
                x1 = pdata1['xp'][ibatch][particleSize][:,0]
                y0 = pdata0['xp'][ibatch][particleSize][:,1]
                y1 = pdata1['xp'][ibatch][particleSize][:,1]
                z0 = pdata0['xp'][ibatch][particleSize][:,2]
                z1 = pdata1['xp'][ibatch][particleSize][:,2]
                dist = ((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)**0.5
                total_dist += dist
    return(total_dist)

def PDF (filename, extension, particleSize):
    distance = totalDist(path + filename, extension, particleSize)
    loc = st.mean(distance)
    scale = np.std(distance)
    y = (distance - loc) / scale
    pdf = norm.pdf(y)
    return (pdf)

def plotPDF (filename, extension, particleSize):
    a = totalDist(path + filename, extension, particleSize)
    b = PDF(path + filename, extension, particleSize)
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.figure(1)
    plt.plot(a,b,'o')
    plt.xlabel('distance')
    plt.ylabel('PDF')
    plt.savefig('distance PDF_' + str(particleSize) + '.png', dpi = 300)
    # plt.show()

def createVideo(name, file):
    cmd1 = 'ffmpeg -f image2 -r 30 -pattern_type glob -i '
    cmd3 = ' -pix_fmt yuv420p -vf '
    cmd4 = '"pad=ceil(iw/2)*2:ceil(ih/2)*2"'
    cmd5 = ' '
    cmd6 = '.mp4'
    cmd = cmd1 + file + cmd3 + cmd4 + cmd5 + name + cmd6
    os.system(cmd)

def stuck (case_name, file_ext):
    #reading data  
    f1 = open( case_name + '.' + file_ext, 'r')
    # n_p = int(f1.readline())  #counts number of probes (reads first line in file)
    f1.close()
    
    #data proccesing
    data = np.genfromtxt( case_name + '.' + file_ext, skip_header = 0, invalid_raise = False)
    num_rows, num_cols = data.shape
    
    hot_wall = [data[i,:] for i in range (0, num_rows) if data[i,3] == -0.5]
    cold_wall = [data[i,:] for i in range (0, num_rows) if data[i,3] == 0.5]
    cold_ceiling = [data[i,:] for i in range (0, num_rows) if data[i,4] == 0.5]
    hot_floor = [data[i,:] for i in range (0, num_rows) if data[i,4] == -0.5]
    adiabatic_front = [data[i,:] for i in range (0, num_rows) if data[i,5] == 0.5]
    adiabatic_back = [data[i,:] for i in range (0, num_rows) if data[i,5] == -0.5]
    
    hot_wall = np.array(hot_wall)
    cold_wall = np.array(cold_wall)     
    cold_ceiling = np.array(cold_ceiling)
    hot_floor = np.array(hot_floor)
    adiabatic_front = np.array(adiabatic_front)
    adiabatic_back = np.array(adiabatic_back)
    
    # plot hot wall
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.figure(1)
    plt.title('hot wall')
    plt.plot(hot_wall[:,4],hot_wall[:,5],'ro')
    #plt.legend()
    plt.xlabel('Z')
    plt.ylabel('Y')
    plt.ylim(-0.5,0.5)
    plt.xlim(-0.5,0.5)
    plt.savefig('hot wall.png', dpi = 300)
    # plt.show()
    
    # plot cold wall
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.figure(1)
    plt.title('cold wall')
    plt.plot(cold_wall[:,4],cold_wall[:,5],'bo',markersize=1)
    #plt.legend()
    plt.xlabel('Z')
    plt.ylabel('Y')
    plt.ylim(-0.5,0.5)
    plt.xlim(-0.5,0.5)
    plt.savefig('cold wall.png', dpi = 300)
    # plt.show()
    
    # plot cold ceiling
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.figure(1)
    plt.title('cold ceiling')
    plt.plot(cold_ceiling[:,3],cold_ceiling[:,5],'bo',markersize=1)
    #plt.legend()
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.ylim(-0.5,0.5)
    plt.xlim(-0.5,0.5)
    plt.savefig('cold ceiling.png', dpi = 300)
    # plt.show()
    
    # plot hot floor
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.figure(1)
    plt.title('hot floor')
    plt.plot(hot_floor[:,3],hot_floor[:,5],'ro',markersize=1)
    #plt.legend()
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.ylim(-0.5,0.5)
    plt.xlim(-0.5,0.5)
    plt.savefig('hot floor.png', dpi = 300)
    # plt.show()

    # plot adiabatic front
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.figure(1)
    plt.title('adiabatic front')
    plt.plot(adiabatic_front[:,3],adiabatic_front[:,4],'o',markersize=1)
    #plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.ylim(-0.5,0.5)
    plt.xlim(-0.5,0.5)
    plt.savefig('adiabatic front.png', dpi = 300)
    # plt.show()
    
    # plot adiabatic back
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.figure(1)
    plt.title('adiabatic back')
    plt.plot(adiabatic_back[:,3],adiabatic_back[:,4],'o',markersize=1)
    #plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.ylim(-0.5,0.5)
    plt.xlim(-0.5,0.5)
    plt.savefig('adiabatic back.png', dpi = 300)
    # plt.show()

def binner(x,y,z,n):
    ii = m.floor((x + 0.5)/(1/n))+1 # 1D x bin index
    jj = m.floor((y + 0.5)/(1/n))+1 # 1D y bin index
    kk = m.floor((z + 0.5)/(1/n))+1 # 1D z bin index
    ind = (ii + (kk-1) * n + (jj-1) * n**2) - 1
    return ind

def matrix(data_t1, data_t2, n):
    A = np.zeros((n**3, n**3))

    nn = np.asarray([n] * len(data_t1))
    
    pt_box_ind_t1 = *map(binner, data_t1[:,0], data_t1[:,2], data_t1[:,1], nn),
    pt_box_ind_t1 = np.asarray(pt_box_ind_t1)
    
    pt_box_ind_t2 = *map(binner, data_t2[:,0], data_t2[:,2], data_t2[:,1], nn),
    pt_box_ind_t2 = np.asarray(pt_box_ind_t2)
    
    for i in range(len(data_t1)):
        A[pt_box_ind_t1[i],pt_box_ind_t2[i]] = A[pt_box_ind_t1[i],pt_box_ind_t2[i]] + 1
    
    return A

def build_matrix (choose, tt1, tt2, n, path1, path2, fileList):
    #params
    num_ps = 5
    x0 = -0.5
    y0 = -0.5
    z0 = -0.5
    
    radius = 0.1
    t1, a1 = particleCoordsNew (path1, fileList[tt1])
    try:
        t2, a2 = particleCoordsNew (path2, fileList[tt2])
    except IndexError:
        t2, a2 = particleCoordsNew (path2, fileList[-1])

    t1 = np.round((t1 - 0.1628499834108E+03), 3)
    t2 = np.round((t2 - 0.1628499834108E+03), 3)
    
    box_coords = [[0 for x in range(n+1)] for x in range(3)]
    box_node = [[0 for x in range(n)] for x in range(3)]
    box_coords[0][0] = x0
    box_coords[1][0] = y0
    box_coords[2][0] = z0
    
    delta = 1/n
    B  = np.zeros((n**3, n**3))
    for j in range(0,3):
        for i in range(1,n+1):
            box_coords[j][i] = box_coords[j][i-1] + delta
            box_node[j][i-1] = (box_coords[j][i-1] + box_coords[j][i])/2
    
    box_coords = np.asarray(np.transpose(box_coords))
    box_node = np.asarray(np.transpose(box_node))
    center_list = (list(itertools.product(box_node[:,0], box_node[:,1], box_node[:,2])))
    center_list = [  np.round(elem,2) for elem in center_list]
    if choose.lower() in ['s', 'sphere']:
        t0, a0 = particleCoordsNew (path1, fileList[0])
        for k in range(len(center_list)):
            print(k-125)
            filtered = []
            ps_index = []
            len_filtered = []
            fff = np.zeros(5)
            center = center_list[k]
            for ps in range(num_ps):
                aa0 = np.asarray(a0[ps])
                for j in range(len(aa0)):
                    r1 = ((aa0[j,0] - center[0])**2 + (aa0[j,1] - center[1])**2 + (aa0[j,2] - center[2])**2)**0.5
                    if r1 <= radius:
                        filtered.append(aa0[j,:])
                len_filtered.append(len(filtered))
                if len(len_filtered) == 1:
                    fff[ps] = len_filtered[ps]
                else:
                    fff[ps] = len_filtered[ps] - len_filtered[ps-1]
                fff = fff.astype(int)
                ps_filtered = []
                count = 0
                for size in (fff):
                    ps_filtered.append([filtered[i+count] for i in range(size)])
                    count += size
                index = np.where(np.isin(aa0[:,1], np.asarray(ps_filtered[ps])))
                ps_index.append(index[0])
            
            data_list1 = []
            data_list2 = []
            for ps in range(num_ps):
                a_np1 = np.asarray(a1[ps])
                data1 = a_np1[ps_index[ps]]
                data_list1.append(data1)
                a_np2 = np.asarray(a2[ps])
                data2 = a_np2[ps_index[ps]]
                data_list2.append(data2)
            
            data_t1 = data_list1[0]
            data_t2 = data_list2[0]
            for ps in range(1,num_ps):
                data_t1 = np.vstack((data_t1, data_list1[ps]))
                data_t2 = np.vstack((data_t2, data_list2[ps]))
                          
            A = matrix(data_t1, data_t2, n)
            
            b = np.argwhere(A[:,:] > 0)   
            B[b[0,0]] = A[b[0,0]]
    elif choose.lower() in ['a', 'all']:
        data_list1 = []
        data_list2 = []
        for ps in range(5):
            a_np1 = np.asarray(a1[ps])
            data_list1.append(a_np1)
            a_np2 = np.asarray(a2[ps])
            data_list2.append(a_np2)
            
        data_t1 = data_list1[0]
        data_t2 = data_list2[0]
        for ps in range(1,5):
            data_t1 = np.vstack((data_t1, data_list1[ps]))
            data_t2 = np.vstack((data_t2, data_list2[ps]))
        
            B = matrix(data_t1, data_t2, n)			
    else:
        print('wrong input!')
    return t1, t2, B

def listfile(folders, file):
    allFileList = []
    listOfFileList = []
    for folder in folders:
        fileList = [name for name in listdir(folder) if name.endswith(file)]
        fileList.sort()
        listOfFileList.append(fileList)
        allFileList = allFileList + fileList
    return (listOfFileList, allFileList)

def find_in_list_of_list(mylist, char):
    for sub_list in mylist:
        if char in sub_list:
            return (mylist.index(sub_list), sub_list.index(char))
    raise ValueError("'{char}' is not in list".format(char = char))
    
def fast_scandir(dirname):
    subfolders= [f.path for f in scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders
