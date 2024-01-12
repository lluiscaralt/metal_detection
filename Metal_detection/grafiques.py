# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:18:58 2024

@author: lluis
"""

import numpy as np
import matplotlib.pyplot as plt
import os 
#%%
"0. Definim el directori inicial"
dir_inicial = os.getcwd()

#%%
"1. Obrim els array's de dades"
os.chdir(dir_inicial+"\\"+"mitjanes_0")
fitxers_0 = os.listdir(os.getcwd())

metal_o_0 = np.load(fitxers_0[0], allow_pickle=True)
metal_x_0 = np.load(fitxers_0[1], allow_pickle=True)
plastic_o_0 = np.load(fitxers_0[2], allow_pickle=True)
plastic_x_0 = np.load(fitxers_0[3], allow_pickle=True)

os.chdir(dir_inicial+"\\"+"mitjanes_1")
fitxers_1= os.listdir(os.getcwd())

metal_o_1 = np.load(fitxers_1[0], allow_pickle=True)
metal_x_1 = np.load(fitxers_1[1], allow_pickle=True)
plastic_o_1 = np.load(fitxers_1[2], allow_pickle=True)
plastic_x_1 = np.load(fitxers_1[3], allow_pickle=True)

os.chdir(dir_inicial+"\\"+"mitjanes_2")
fitxers_2= os.listdir(os.getcwd())

metal_o_2 = np.load(fitxers_2[0], allow_pickle=True)
metal_x_2 = np.load(fitxers_2[1], allow_pickle=True)
plastic_o_2 = np.load(fitxers_2[2], allow_pickle=True)
plastic_x_2 = np.load(fitxers_2[3], allow_pickle=True)

provon = metal_o_0[1, :]
caca = metal_o_0[1, :][0]

#%%
"2. Definim una funció per processar les dades del 3D- plot"
def dades_3D(data):
    s1 = []
    s2 = []
    s3 = []
    for i in range (np.shape(data)[1]):
        s1.append(data[1, i][0])
        s2.append(data[1, i][1])
        s3.append(data[1, i][2])
    return (s1, s2, s3)


#%%
"3. Representació 3D de s1, s2 i s3 pel medi 0"
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(dades_3D(metal_o_0)[0], dades_3D(metal_o_0)[1], dades_3D(metal_o_0)[2], color = "#F00B1D", label = "metal valid 0")
ax.scatter(dades_3D(metal_x_0)[0], dades_3D(metal_x_0)[1], dades_3D(metal_x_0)[2], color = "#F39EA5", label = "metal no valid 0", marker="x")
ax.scatter(dades_3D(plastic_o_0)[0], dades_3D(plastic_o_0)[1], dades_3D(plastic_o_0)[2], color = "#06D7F4", label = "plastic valid 0")
ax.scatter(dades_3D(plastic_x_0)[0], dades_3D(plastic_x_0)[1], dades_3D(plastic_x_0)[2], color = "#ACEFF7", label = "plastic no valid 0", marker ="x")

ax.set_xlabel(r"$s_1$", fontsize = 14)
ax.set_ylabel(r"$s_2$", fontsize = 14)
ax.set_zlabel(r"$s_3$", fontsize = 14)

ax.set_xticks(np.arange(-1, 1.2, 0.4))
ax.set_yticks(np.arange(-1, 1.2, 0.4))
ax.set_zticks(np.arange(-1, 1.2, 0.4))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=2, fancybox=True, shadow=True)


plt.tight_layout()


plt.show()

#%%
"4. Tractament de les dades 2D plots"
def tractament_dades(data):
    dades_L = []
    dades_C = []
    for i in range (np.shape(data)[1]):
        dades_L.append(np.sqrt((data[1, i][0])**2+(data[1, i][1])**2))
        dades_C.append(data[1, i][2])
    return (dades_L, dades_C)


PL_m_o_0 = tractament_dades(metal_o_0)[0]
PL_m_x_0 = tractament_dades(metal_x_0)[0]
PL_p_o_0 = tractament_dades(plastic_o_0)[0]
PL_p_x_0 = tractament_dades(plastic_x_0)[0]
PC_m_o_0 = tractament_dades(metal_o_0)[1]
PC_m_x_0 = tractament_dades(metal_x_0)[1]
PC_p_o_0 = tractament_dades(plastic_o_0)[1]
PC_p_x_0 = tractament_dades(plastic_x_0)[1]

PL_m_o_1 = tractament_dades(metal_o_1)[0]
PL_m_x_1 = tractament_dades(metal_x_1)[0]
PL_p_o_1 = tractament_dades(plastic_o_1)[0]
PL_p_x_1 = tractament_dades(plastic_x_1)[0]
PC_m_o_1 = tractament_dades(metal_o_1)[1]
PC_m_x_1 = tractament_dades(metal_x_1)[1]
PC_p_o_1 = tractament_dades(plastic_o_1)[1]
PC_p_x_1 = tractament_dades(plastic_x_1)[1]

PL_m_o_2 = tractament_dades(metal_o_2)[0]
PL_m_x_2 = tractament_dades(metal_x_2)[0]
PL_p_o_2 = tractament_dades(plastic_o_2)[0]
PL_p_x_2 = tractament_dades(plastic_x_2)[0]
PC_m_o_2 = tractament_dades(metal_o_2)[1]
PC_m_x_2 = tractament_dades(metal_x_2)[1]
PC_p_o_2 = tractament_dades(plastic_o_2)[1]
PC_p_x_2 = tractament_dades(plastic_x_2)[1]

#%%
"5. Representació"

plt.figure()

plt.scatter(PL_m_o_0, PC_m_o_0, color = "#F00B1D", label = "metal valid 0")
for i, label in enumerate(metal_o_0[0]):
    plt.annotate(label, (PL_m_o_0[i], PC_m_o_0[i]))
plt.scatter(PL_m_x_0, PC_m_x_0, color = "#F39EA5", label = "metal no valid 0", marker="x")
for i, label in enumerate(metal_x_0[0]):
    plt.annotate(label, (PL_m_x_0[i], PC_m_x_0[i]))

plt.scatter(PL_p_o_0, PC_p_o_0, color = "#06D7F4", label = "plastic valid 0")
for i, label in enumerate(plastic_o_0[0]):
    plt.annotate(label, (PL_p_o_0[i], PC_p_o_0[i]))
plt.scatter(PL_p_x_0, PC_p_x_0, color = "#ACEFF7", label = "plastic no valid 0", marker ="x")
for i, label in enumerate(plastic_x_0[0]):
    plt.annotate(label, (PL_p_x_0[i], PC_p_x_0[i]))

plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
plt.xlabel (r"$P_L$")
plt.xlim(0, 1.4)
plt.xticks(np.arange(0, 1.5, 0.2))
plt.ylabel (r"$P_C$")
plt.ylim(-1, 1)
plt.yticks(np.arange(-1, 1.2, 0.2))
plt.grid()
plt.tight_layout()
plt.show()

plt.figure()

plt.scatter(PL_m_o_1, PC_m_o_1, color = "#F00B1D", label = "metal valid 1")
for i, label in enumerate(metal_o_1[0]):
    plt.annotate(label, (PL_m_o_1[i], PC_m_o_1[i]))
plt.scatter(PL_m_x_1, PC_m_x_1, color = "#F39EA5", label = "metal no valid 1", marker ="x")
for i, label in enumerate(metal_x_1[0]):
    plt.annotate(label, (PL_m_x_1[i], PC_m_x_1[i]))

plt.scatter(PL_p_o_1, PC_p_o_1, color = "#06D7F4", label = "plastic valid 1")
for i, label in enumerate(plastic_o_1[0]):
    plt.annotate(label, (PL_p_o_1[i], PC_p_o_1[i]))
plt.scatter(PL_p_x_1, PC_p_x_1, color = "#ACEFF7", label = "plastic no valid 1", marker ="x")
for i, label in enumerate(plastic_x_1[0]):
    plt.annotate(label, (PL_p_x_1[i], PC_p_x_1[i]))
    
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
plt.xlabel (r"$P_L$")
plt.xlim(0, 1.4)
plt.xticks(np.arange(0, 1.5, 0.2))
plt.ylabel (r"$P_C$")
plt.ylim(-1, 1)
plt.yticks(np.arange(-1, 1.2, 0.2))
plt.grid()
plt.tight_layout()
plt.show()

plt.figure()

plt.scatter(PL_m_o_2, PC_m_o_2, color = "#F00B1D", label = "metal valid 2")
for i, label in enumerate(metal_o_2[0]):
    plt.annotate(label, (PL_m_o_2[i], PC_m_o_2[i]))
plt.scatter(PL_m_x_2, PC_m_x_2, color = "#F39EA5", label = "metal no valid 2", marker ="x")
for i, label in enumerate(metal_x_2[0]):
    plt.annotate(label, (PL_m_x_2[i], PC_m_x_2[i]))

plt.scatter(PL_p_o_2, PC_p_o_2, color = "#06D7F4", label = "plastic valid 2")
for i, label in enumerate(plastic_o_2[0]):
    plt.annotate(label, (PL_p_o_2[i], PC_p_o_2[i]))
plt.scatter(PL_p_x_2, PC_p_x_2, color = "#ACEFF7", label = "plastic no valid 2", marker ="x")
for i, label in enumerate(plastic_x_2[0]):
    plt.annotate(label, (PL_p_x_2[i], PC_p_x_2[i]))



plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
plt.xlabel (r"$P_L$")
plt.xlim(0, 1.4)
plt.xticks(np.arange(0, 1.5, 0.2))
plt.ylabel (r"$P_C$")
plt.ylim(-1, 1)
plt.yticks(np.arange(-1, 1.2, 0.2))
plt.grid()
plt.tight_layout()

plt.show()
#%%
"6. Gràfica vectorial"
x_metal = []
y_metal = []
U_raw = []
V_raw = []
for i in range (np.shape(metal_o_0)[1]):
    metal = np.array([])
    anot = metal_o_0[0][i]
    metal = np.append(metal, anot)
    metal = np.append(metal, np.where(metal_o_1[0] == anot ))
    if np.shape(metal)[0]== 2:
        x_metal.append(PL_m_o_0[i])
        y_metal.append(PC_m_o_0[i])
        U_raw.append(PL_m_o_1[int(metal[1])])
        V_raw.append(PC_m_o_1[int(metal[1])])
        
U = np.array(U_raw)-np.array(x_metal)
V = np.array(V_raw)-np.array(y_metal)     

#%%
"7. Definim una funció per tal de realitzar els vectors en les gràfiques"
def vectors (metal_plastic, metal_plastic_mes1, PL_metal_plastic, PC_metal_plastic, PL_metal_plastic_mes1, PC_metal_plastic_mes1):
    x = []
    y = []
    U_raw = []
    V_raw = []
    etiquetes = []
    for i in range (np.shape(metal_plastic)[1]):
        comprov = np.array([])
        anot = metal_plastic[0][i]
        comprov = np.append(comprov, anot)
        comprov = np.append(comprov, np.where(metal_plastic_mes1[0] == anot ))
        if np.shape(comprov)[0]== 2:
            etiquetes.append(anot)
            x.append(PL_metal_plastic[i])
            y.append(PC_metal_plastic[i])
            U_raw.append(PL_metal_plastic_mes1[int(comprov[1])])
            V_raw.append(PC_metal_plastic_mes1[int(comprov[1])])
    
    
    x = np.array(x)
    y = np.array(y)
    U_raw = np.array(U_raw)
    V_raw = np.array(V_raw)
    U = U_raw-x
    V = V_raw-y
    etiquetes = np.array(etiquetes)
    return (x, y, U, V, U_raw, V_raw, etiquetes)
#%%
"8. Representem la gràfica vectorial per tots els casos"
"8.1. Cas sense medi dispersiu i medi dispersiu 1"
data_vect = vectors(metal_o_0, metal_o_1, PL_m_o_0, PC_m_o_0, PL_m_o_1, PC_m_o_1)

plt.figure()
plt.quiver(data_vect[0], data_vect[1], data_vect[2], data_vect[3], angles='xy', scale_units='xy', scale=1, 
           linestyle = "dashed", width = 0.0015)
plt.scatter(data_vect[0], data_vect[1], color = "#F00B1D", label = "metal valid 0")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[0][i], (data_vect[1][i])))


plt.scatter(data_vect[4], data_vect[5], color = "#392222", label = "metal valid 1")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[4][i], (data_vect[5][i])))

data_vect = vectors(plastic_o_0, plastic_o_1, PL_p_o_0, PC_p_o_0, PL_p_o_1, PC_p_o_1)

plt.quiver(data_vect[0], data_vect[1], data_vect[2], data_vect[3], angles='xy', scale_units='xy', scale=1, 
           linestyle = "dashed", width = 0.0015)
plt.scatter(data_vect[0], data_vect[1], color = "#06D7F4", label = "plastic valid 0")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[0][i], (data_vect[1][i])))


plt.scatter(data_vect[4], data_vect[5], color = "#2675C8", label = "plastic valid 1")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[4][i], (data_vect[5][i])))

plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
plt.xlabel (r"$P_L$")
plt.xlim(0, 1.4)
plt.xticks(np.arange(0, 1.5, 0.2))
plt.ylabel (r"$P_C$")
plt.ylim(-1, 1)
plt.yticks(np.arange(-1, 1.2, 0.2))
plt.grid()
plt.tight_layout()

plt.show()

#%%
"8.2. Cas sense medi dispersiu i medi dispersiu 2"
data_vect = vectors(metal_o_0, metal_o_2, PL_m_o_0, PC_m_o_0, PL_m_o_2, PC_m_o_2)

plt.figure()
plt.quiver(data_vect[0], data_vect[1], data_vect[2], data_vect[3], angles='xy', scale_units='xy', scale=1, 
           linestyle = "dashed", width = 0.0015)
plt.scatter(data_vect[0], data_vect[1], color = "#F00B1D", label = "metal valid 0")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[0][i], (data_vect[1][i])))


plt.scatter(data_vect[4], data_vect[5], color = "#392222", label = "metal valid 2")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[4][i], (data_vect[5][i])))

data_vect = vectors(plastic_o_0, plastic_o_2, PL_p_o_0, PC_p_o_0, PL_p_o_2, PC_p_o_2)

plt.quiver(data_vect[0], data_vect[1], data_vect[2], data_vect[3], angles='xy', scale_units='xy', scale=1, 
           linestyle = "dashed", width = 0.0015)
plt.scatter(data_vect[0], data_vect[1], color = "#06D7F4", label = "plastic valid 0")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[0][i], (data_vect[1][i])))


plt.scatter(data_vect[4], data_vect[5], color = "#2675C8", label = "plastic valid 2")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[4][i], (data_vect[5][i])))

plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
plt.xlabel (r"$P_L$")
plt.xlim(0, 1.4)
plt.xticks(np.arange(0, 1.5, 0.2))
plt.ylabel (r"$P_C$")
plt.ylim(-1, 1)
plt.yticks(np.arange(-1, 1.2, 0.2))
plt.grid()
plt.tight_layout()

plt.show()

#%%
"8.3. Cas medi dispersiu 1 i medi dispersiu 2"
data_vect = vectors(metal_o_1, metal_o_2, PL_m_o_1, PC_m_o_1, PL_m_o_2, PC_m_o_2)

plt.figure()
plt.quiver(data_vect[0], data_vect[1], data_vect[2], data_vect[3], angles='xy', scale_units='xy', scale=1, 
           linestyle = "dashed", width = 0.0015)
plt.scatter(data_vect[0], data_vect[1], color = "#F00B1D", label = "metal valid 1")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[0][i], (data_vect[1][i])))


plt.scatter(data_vect[4], data_vect[5], color = "#392222", label = "metal valid 2")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[4][i], (data_vect[5][i])))

data_vect = vectors(plastic_o_1, plastic_o_2, PL_p_o_1, PC_p_o_1, PL_p_o_2, PC_p_o_2)

plt.quiver(data_vect[0], data_vect[1], data_vect[2], data_vect[3], angles='xy', scale_units='xy', scale=1, 
           linestyle = "dashed", width = 0.0015)
plt.scatter(data_vect[0], data_vect[1], color = "#06D7F4", label = "plastic valid 1")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[0][i], (data_vect[1][i])))


plt.scatter(data_vect[4], data_vect[5], color = "#2675C8", label = "plastic valid 2")
for i, label in enumerate(data_vect[6]):
    plt.annotate(label, (data_vect[4][i], (data_vect[5][i])))

plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
plt.xlabel (r"$P_L$")
plt.xlim(0, 1.4)
plt.xticks(np.arange(0, 1.5, 0.2))
plt.ylabel (r"$P_C$")
plt.ylim(-1, 1)
plt.yticks(np.arange(-1, 1.2, 0.2))
plt.grid()
plt.tight_layout()


plt.show()





