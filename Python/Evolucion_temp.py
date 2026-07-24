############################################### EVOLUCIÓN DE LA TEMPERATURA GRANULAR #############################################################################################
# importar librerías necesarias 
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



# leer los datos para cada config de gas granular
h = pd.read_pickle('/data/jpolobar/granular/filtrado_45.pkl')

f= pd.read_pickle('/data/jpolobar/granular/filtrado_50.pkl')

e = pd.read_pickle('/data/jpolobar/granular/filtrado_60.pkl')

d= pd.read_pickle('/data/jpolobar/granular/filtrado_70.pkl')

# Cálculo de la T_g
T_g = 0.5 * h.groupby('frame')['v2'].mean()
T_g2 = 0.5 * f.groupby('frame')['v2'].mean()
T_g3 = 0.5 * e.groupby('frame')['v2'].mean()
T_g4 = 0.5 * d.groupby('frame')['v2'].mean()


# filtro gaussiano para eliminar ruido
T_g_filtered = gaussian_filter1d(T_g, sigma=60)
T_g2_filtered = gaussian_filter1d(T_g2, sigma=60)
T_g3_filtered = gaussian_filter1d(T_g3, sigma=60)
T_g4_filtered = gaussian_filter1d(T_g4, sigma=70)


# graficar 
plt.figure(figsize=(6.4,4))
plt.scatter((T_g.index / 1000), T_g_filtered, marker='.',s=0.1,label = r'$\Gamma = 4.05$')
plt.scatter((T_g2.index / 1000)-1.2, T_g2_filtered,  marker='.',s=0.1,label = r'$\Gamma = 4.50$') # alinear temporalmente todas las gráficas con respecto a una de las curvas 
plt.scatter((T_g3.index / 1000)-0.8, T_g3_filtered,  marker='.',s=0.1,label = r'$\Gamma = 5.40$')
plt.scatter((T_g4.index / 1000)-0.3, T_g4_filtered,  marker='.',s=0.1,label = r'$\Gamma = 6.30$')
plt.xlim(0,30)
plt.ylim(0,0.04)
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$T_g\,[\sigma^2/\mathrm{s}^2]$')
plt.legend(frameon=False, fontsize=9, loc='best',markerscale=15)
plt.grid(False)
plt.savefig('T_final.png', dpi=600, bbox_inches='tight')
plt.show()



