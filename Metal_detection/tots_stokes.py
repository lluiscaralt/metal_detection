# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:48:31 2023

@author: lluis
"""

import os
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import cv2
from skimage.filters import try_all_threshold
from skimage.filters import threshold_triangle
from skimage.filters import threshold_mean
from scipy.signal import correlate2d
from scipy.ndimage import shift
from scipy.ndimage import binary_erosion
from scipy import ndimage



#%%
"0. Tractament que farem a les imatges dels directoris"
def tractament_cockpit(im_BGR):
    imGL= np.float16((im_BGR[:, :, 2]/16).astype(np.uint16))#Important cv2 ho fa amb BGR
    imGL_def_thresh= imGL>threshold_triangle(imGL) #Fem un threshold
    structure = np.ones((3, 3)) #Definim el "kernel" de l'erosió
    imGL_def = binary_erosion(imGL_def_thresh, structure)*imGL #Fem l'erosió
    return np.array(imGL_def) #Ens quedem només amb el canal vermell

#%%
"1. Funció per alinear les dues imatges a restar mitjançant la correlació creuada en l'espai de Fourier"
def cross_correlation (im_1, im_2):
    corr = np.abs(fft.ifftshift(fft.ifft2(fft.fftshift(fft.fft2(im_1)) *np.conjugate(fft.fftshift(fft.fft2(im_2))))))
    index = np.array (np.where (corr == np.max(corr))).reshape((2, ))

    center = (np.array(np.shape(corr)) //2).reshape(2, )

    trans = center-index

    im1_centrada = ndimage.shift(np.float32(im_1), trans)
    
    return np.float16(im1_centrada)

#%%
"2. Funció que ens permetrà emmagetzemar les fotos amb el valor de cada paràmetre de Stokes píxel a píxel"
stokes_data = []
def stokes_def(data, llista_stokes):
    for i in range (int(np.shape(data)[0]/2)):
        data_1_raw = data[i*2]
        data_2 = data[i*2+1]
        data_1 = cross_correlation (data_1_raw, data_2)
        mask = (data_1==0)*(data_2==0)
        
        dades_1_def = np.ma.masked_array(data_1, mask)
        dades_2_def = np.ma.masked_array(data_2, mask)
        S_raw = (dades_1_def - dades_2_def)/(dades_1_def + dades_2_def)
        S = np.isfinite(S_raw)*S_raw 
        stokes_data.append (S)
    return stokes_data
    
#%%
"3. Funció que ens permetrà fer la mitjana de cada imatge de Stokes i emmegatzemar els valors"
mitjana_stokes = []
def mitjana_stokes_calc(data_stokes):
    for i in range (np.shape(data_stokes)[0]):
        mitjana_stokes.append(np.ma.mean(np.float64(data_stokes[i])))
    
    return mitjana_stokes
        
#%%
"4. Funció que ens obrirà totes les imatges del directori i en farà el tractament corresponent"

imatges_ref = [] #Definim un array on hi col·locarem una imatge de referència per mostrar al subplots
def dades (directori):
    os.chdir (directori)
    arxius = os.listdir(directori)
    dades = []
    im_BGR_ref = cv2.imread(arxius[0], cv2.IMREAD_UNCHANGED)
    im_RGB_ref = np.ones(np.shape(im_BGR_ref))
    im_RGB_ref [:, :, 0] = np.float16((im_BGR_ref[:, :, 2]/16).astype(np.uint16))
    im_RGB_ref [:, :, 1] = np.float16((im_BGR_ref[:, :, 1]/16).astype(np.uint16))
    im_RGB_ref [:, :, 2] = np.float16((im_BGR_ref[:, :, 0]/16).astype(np.uint16))
    imatges_ref.append(im_RGB_ref/np.max(im_RGB_ref))
    for name in arxius:
        im_GBR = cv2.imread(str(name), cv2.IMREAD_UNCHANGED) #Fem el tractament imatge per imatge
        dades.append(tractament_cockpit(im_GBR))
    
    return (np.array(dades))

#%%
"5. Canviem de directori per obrir les imatges del set d'objectes i definim el directori inicial"
dir_inicial = os.getcwd()
os.chdir (dir_inicial+str("\\metal")) #Canviar en funció del directori
dir_origen = os.getcwd()
objectes = os.listdir(dir_origen)

#%%
"6. Cridem a la funció dades i stokes_def per cada objecte"
for objecte in objectes:
    data = dades (str(dir_origen)+str("\\")+str(objecte)) #Cridem a la funció dades pel directori de treball
    stokes_def(data, stokes_data)

#%%
"7. Fem la mitjana de stokes_data"
mitjana = mitjana_stokes_calc(stokes_data)
#%%
"8. Represnetem per cada objecte una imatge de referència, una imatge de $s_1$, $s_2$ i $s_3$"
for i in range (np.shape(objectes)[0]):
    
    plt.figure()
    
    plt.subplot(1, 4, 1)
    plt.imshow(imatges_ref[i])
    plt.axis("off")
    
    plt.subplot(1, 4, 2) #float64 per tal d'evitar l'overflow
    plt.title("Mean $s_1$ = {:.3f}" .format(mitjana[i*3]), fontsize=10)
    plt.imshow(stokes_data[i*3], cmap = "seismic")
    plt.axis("off")
    colorbar = plt.colorbar(shrink = 0.2, format='%.0f')
    colorbar.ax.tick_params(labelsize = "small")
    
    plt.subplot(1, 4, 3)
    plt.title("Mean $s_3$ = {:.3f}" .format(mitjana[i*3+1]), fontsize=10)
    plt.imshow(stokes_data[i*3+1], cmap = "seismic", )
    plt.axis("off")
    colorbar = plt.colorbar(shrink = 0.2, format='%.0f')
    colorbar.ax.tick_params(labelsize = "small")
    
    plt.subplot(1, 4, 4)
    plt.title("Mean $s_3$ = {:.3f}" .format(mitjana[i*3+2]), fontsize=10)
    plt.imshow(stokes_data[i*3+2], cmap = "seismic")
    plt.axis("off")
    colorbar = plt.colorbar(shrink = 0.2, format='%.0f')
    colorbar.ax.tick_params(labelsize = "small")
    
    plt.tight_layout()
    
    
    plt.savefig(str(dir_inicial)+"\\stokes_metal"+str("\\")+str(i+1)+".png", dpi = 1000, bbox_inches='tight', pad_inches=0.1) #Canviar en funció del directori
    #Canviar en funció del directori 

#%%
"8. Funció que ens permet discernir si els objectes són o no significatius"
def data_out (data_stokes):
    annotate_o = [] #Etiquetes significatius
    annotate_x = [] #Etiquetes no significatius
    mean_o = [] #Mitjanes significatius
    mean_x = [] #Mitjanes no significatius
    for i in range(int(np.shape(data_stokes)[0]/3)):
        mean = []
        mean.append(np.ma.mean(np.float64(stokes_data[i*3])))
        mean.append(np.ma.mean(np.float64(stokes_data[i*3+1])))
        mean.append(np.ma.mean(np.float64(stokes_data[i*3+2])))
        if np.sum(data_stokes[i*3]!=np.float16(0))>=(0.01*(1944*2592)):
            annotate_o.append(i+1)
            mean_o.append(mean)

        else:
            annotate_x.append(i+1)
            mean_x.append(mean)
        
    return ([annotate_o, mean_o], [annotate_x, mean_x])

output_data = data_out(stokes_data)

#%%
"9. Guardem les dades que després representarem"
np.save(str(dir_inicial)+"\\mitjanes_0\\metal_o.npy", np.array(output_data[0], dtype = "object")) #Canviar de directori
np.save(str(dir_inicial)+"\\mitjanes_0\\metal_x.npy", np.array(output_data[1], dtype = "object")) #Canviar de directori
#Canviar en funció del directori i del tipus de metarial a guardar (metal o plastic)
