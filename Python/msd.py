########## CALCULAR MSD ###############
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
from scipy.integrate import trapezoid
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



fase1_inicio = 6000 # inicio vibración aprox
fase1_fin = 26000 # fin vibración aprox

sigma_px = 6.55  # diámetro en pixeles

# Leer y filtrar directamente
fase1 = pd.read_hdf('/data5TB/jpolobar/granular/trayectorias_gas45.h5',
                     where='frame >= fase1_inicio & frame <= fase1_fin')
fase2 = pd.read_hdf('/data5TB/jpolobar/granular/trayectorias_gas60.h5',
                     where='frame >= fase1_inicio & frame <= fase1_fin')

fase3 = pd.read_hdf('/data5TB/jpolobar/granular/trayectorias_gas70.h5',
                     where='frame >= fase1_inicio & frame <= fase1_fin')



# MSD en pixeles^2
em = tp.emsd(fase1, mpp=1, fps=1000)
em2 = tp.emsd(fase2, mpp=1, fps=1000)
em3 = tp.emsd(fase3, mpp=1, fps=1000)

# Convertir a unidades de sigma^2
em_sigma = em / sigma_px**2
em_sigma2 = em2 / sigma_px**2
em_sigma3 = em3 / sigma_px**2

# Graficar
plt.figure(figsize=(6.4,4))
plt.plot(em_sigma.index, em_sigma,lw=1.5, marker='o',markerfacecolor='white', linestyle='-', markersize=4, label=r'$\Gamma = 4.05$')
plt.plot(em_sigma2.index, em_sigma2,lw=1.5, marker='s',markerfacecolor='white', linestyle='-', markersize=4, label=r'$\Gamma = 5.40$')
plt.plot(em_sigma3.index, em_sigma3,lw=1.5, color='#467821',marker='^',markerfacecolor='white', linestyle='-', markersize=4, label=r'$\Gamma = 6.30$')

# Recta de Difusión Normal 
t_dif = np.array([0.06, 0.1]) 
# Ajustamos la altura (el prefactor) para que quede cerca de la curva superior
msd_dif = 16* t_dif**1  
plt.plot(t_dif, msd_dif, 'k:', lw=1)
plt.text(0.07, 0.7, r'$\sim t$', fontsize=10)

# Recta de Régimen Balístico 
t_bal = np.array([0.0012, 0.002])
# Ajustamos la altura
msd_bal = 1500 * t_bal**2 
plt.plot(t_bal, msd_bal, 'k:', lw=1)
plt.text(0.0018, 0.003, r'$\sim t^2$', fontsize=10)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.00095,0.105)
plt.ylim(0.002,4.1)
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$\langle \Delta r^2 (t)\rangle / \sigma^2$')
plt.legend(frameon=False, fontsize=9, loc='best')
plt.grid(False)
plt.tight_layout()
plt.savefig('msd.png', dpi=600, bbox_inches='tight')
plt.show()
