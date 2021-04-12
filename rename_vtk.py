#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:09:18 2021

@author: akimlavrinenko
"""

import os
import numpy as np
path = '/Users/akimlavrinenko/Documents/coding/data/OF_ps_data/stochastic_k0_VTK/ps/'
files = os.listdir(path)
files.sort()


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join(['ps_' + str(np.round(((index + 1) * 0.05),3)), '.vtk'])))