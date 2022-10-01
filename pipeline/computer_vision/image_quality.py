'''
Metrics for unferwater image quality evaluation.

Author: Xuelei Chen 
Email: chenxuelei@hotmail.com

Usage:
python evaluate.py RESULT_PATH REFERENCE_PATH
'''
import numpy as np
import math
import sys
from skimage import io, color, filters
import os
import math
from skimage.util import img_as_float


def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:,:,0]

    #1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5

    #2nd term
    top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l,axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[::top])-np.mean(sl[::top])

    #3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0: satur.append(0)
        elif l1[i] == 0: satur.append(0)
        else: satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    return uciqe

def analyse_image(image):
    corrected = img_as_float(image) #io.imread(os.path.join(result_path,imgdir))

    #reference image
    uciqe = nmetrics(corrected)

    return uciqe

