###### TRAYECTORIA DE UNA PARTÍCULA  #############
# librerías
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import cv2
import os
import glob
import gc
import scipy
from scipy.stats import norm
from scipy.spatial.distance import pdist
import trackpy as tp
import pandas as pd
import pims
from pandas import DataFrame, Series

# Configuramos el estilo del gráfico
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 12,
    'figure.figsize': (6.4, 4.0),
    'legend.fontsize': 9
})



cols = ['particle', 'frame', 'x', 'y']
t = pd.read_hdf('/data5TB/jpolobar/granular/trayectorias_gas50.h5', columns=cols)


# Filtrar y ordenar cronológicamente la trayectoria de la partícula
trayectoria = t[
    (t['particle'] == 3955) & 
    (t['frame'] >= 6000) & 
    (t['frame'] <= 26000)
].sort_values('frame')  # trayectoria que dura muchos frames


# graficar:

plt.figure(figsize=(6.4, 4))
S = 1/6.55 #  diámetro por píxel
# Dibujar la línea guía del camino en gris
plt.plot(trayectoria['x']*S, trayectoria['y']*S, color='black', alpha=0.2, linewidth=0.8, zorder=1)

# mapa de calor
sc = plt.scatter(trayectoria['x']*S, trayectoria['y']*S, 
                 c=trayectoria['frame']/1000, cmap='plasma', 
                 s=3, alpha=0.7, zorder=2)

# barra de color lateral para identificar la evolución temporal
cbar = plt.colorbar(sc)
cbar.set_label('t [s]', fontsize=11)

# Destacar con marcadores más grandes el primer y el último punto
plt.scatter(trayectoria['x'].iloc[0]*S, trayectoria['y'].iloc[0]*S, 
            color='green', marker='o', s=50, label='Posición inicial', zorder=3)
plt.scatter(trayectoria['x'].iloc[-1]*S, trayectoria['y'].iloc[-1]*S, 
            color='red', marker='X', s=60, label='Posición final', zorder=3)

plt.xlabel(r'$x/\sigma$', fontsize=12)
plt.ylabel(r'$y/\sigma$', fontsize=12)
plt.xlim(0,20.7)
plt.ylim(0,20)
plt.xticks(np.arange(0,21,5))
plt.yticks(np.arange(0,21,5))
plt.grid(False)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('img/trayectoria.png', dpi=600, bbox_inches='tight')
plt.show()
