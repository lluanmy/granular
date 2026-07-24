###############################ISOTROPÍA DEL GAS ####################################################################
# importar librerías
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import cv2
import os
import glob
import gc
import scipy
from scipy.stats import norm
from scipy.stats import linregress
from scipy.spatial.distance import pdist
import trackpy as tp
import pandas as pd
import pims
from pandas import DataFrame, Series
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


# Configurar el estilo del gráfico
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





fase1_inicio = 6000 # inicio de la vibración aprox
fase1_fin = 26000 # # fin de la vibración aprox

# leer datos
f = pd.read_hdf('/data/jpolobar/granular/filtrado_gas.h5',
                where='frame >= fase1_inicio & frame <= fase1_fin')


fig, axs = plt.subplots(figsize=(6,6))
plt.subplots_adjust(wspace=0.3)

# velocidades
v1 = fase1['vx'].dropna().values 
v2 = fase1['vy'].dropna().values 


# Calcular velocidad característica v0
v01 = np.sqrt(2*np.mean(v1**2))   
v02 = np.sqrt(2*np.mean(v2**2)) 

# Escalar
v_scaled1 = v1 / v01
v_scaled2 = v2 / v02



axs.scatter(v_scaled1, v_scaled2, s=3, alpha=0.3, color='black',label=r'$\Gamma = 5.51$')
axs.set_xlim(-20,20)
axs.set_ylim(-20,20)
axs.set_xlabel(r'$v_x/v_0$', fontsize=12)
axs.set_ylabel(r'$v_y/v_0$', fontsize=12)
axs.grid(alpha=0.3)
axs.legend(frameon=False, fontsize=9, loc='best')
plt.tight_layout()
plt.grid(False)
plt.savefig('img/cristal_scatter_vel.png', dpi=600, bbox_inches='tight')
plt.show()
