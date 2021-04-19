#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:47:20 2021

@author: akimlavrinenko
"""

from openFoamClass import openFoam


dirPath = '/Users/akimlavrinenko/Documents/coding/data/OF_ps_data/test0_k0'
foldName0 = 'reactingCloud1'
foldName1 = 'posititions_k0'


openFoam().renameLPTData(dirPath, foldName0, 0.05, '.txt')
velocity = openFoam().readLPTVelocity(dirPath, foldName0, 0.05, '.txt')
position = openFoam().readLPTPositions(dirPath, foldName1, 0.05)