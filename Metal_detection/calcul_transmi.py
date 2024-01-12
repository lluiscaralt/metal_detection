# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 23:55:04 2023

@author: lluis
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold
from skimage.filters import threshold_triangle
from skimage.filters import threshold_mean
from scipy.signal import correlate2d
from scipy.ndimage import shift
from scipy.ndimage import binary_erosion
from scipy import ndimage

#%%
"0. Llistem les 3 imatges que tenim en el directori calcul_trasnmi i definim la llista de transmitàncies"
os.chdir (os.getcwd()+str("\\calcul_transmi")) #Canviar en funció del directori
llista_dir = os.listdir(os.getcwd())
transmi = [0, 0, 0]
titols = ["Medium 0", "Medium 1", "Medium 2"]

#%%
"1. Definim una funció per tal d'obtenir el threshold que aplicarem a les 3 imatges"
def tractament_cockpit_thresh(im_BGR):
    imGL= np.float16((im_BGR[:, :, 2]/16).astype(np.uint16))#Important cv2 ho fa amb BGR
    imGL_def_thresh= imGL>threshold_triangle(imGL) #Fem un threshold
    structure = np.ones((3, 3)) #Definim el "kernel" de l'erosió
    imbin_def = binary_erosion(imGL_def_thresh, structure) #Fem l'erosió
    return np.array(imbin_def) #Ens quedem només amb el canal vermell
#%%
"2. Obtenim el threshold que serà el mateix per les 3 imatges a partir de la foto sense boira"
im_1 = cv2.imread(llista_dir[0], cv2.IMREAD_UNCHANGED)
thresh = tractament_cockpit_thresh(im_1)
#%%
"3. Apliquem el threshold a les 3 imatges i calculem "
plt.figure()
for i in range (len(llista_dir)):
    plt.subplot(1, 3, i+1)
    im_BGR_trans = cv2.imread(llista_dir[i], cv2.IMREAD_UNCHANGED)
    im_trans = thresh*np.float16((im_BGR_trans[:, :, 2]/16).astype(np.uint16))
    plt.imshow(im_trans, cmap = "gray")
    plt.axis("off")
    plt.title(str(titols[i]), fontsize = 9)
    transmi[i] = np.sum(np.float64(im_trans))
    plt.tight_layout()
plt.show()
    
#%%
"4. Calculem el valor de la transmitància en tant per cent"
t0 = (transmi[0]/transmi[0])*100
t1 = (transmi[1]/transmi[0])*100
t2 = (transmi[2]/transmi[0])*100
