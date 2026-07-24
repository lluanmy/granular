########### CALCULAR g(r) ####################################
# librerías 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import cv2
import os
import glob
import gc
import scipy
from scipy.stats import norm,kurtosis
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
plt.rcParams.update({'text.usetex': True,'font.family': 'serif', 'font.size': 12})


# Parámetros
fps = 1000
dt = 0.001
sigma = 0.23825   # cm
m = 0.0555         # g
# inicio y fin del forzamiento
fase1_inicio = 6000
fase1_fin = 25000

def calcular_gr(df, frames, dr, D, d_part):
    R = D / 2
    A_total = np.pi * R**2
    S = 25.6 / 704    # cm por píxel

    r_bins = np.arange(0, R + dr, dr) # intervalos
    r_vals = 0.5 * (r_bins[:-1] + r_bins[1:])
    shell_area = 2 * np.pi * r_vals * dr # área del anillo

    # inicializar
    h_total = np.zeros_like(r_vals)
    nframes_valid = 0

    for i, frame in enumerate(frames, 1):
        df_frame = df[df["frame"] == frame]
        N = len(df_frame)
        if N < 2:  # Asegurar (solo se pueden calcular dist entre 2 partículas )
            continue

        nframes_valid += 1  # acumulamos
        
        # Posiciones en cm
        coords = df_frame[["x", "y"]].to_numpy() * S

        dx = coords[:, None, 0] - coords[None, :, 0] # matriz (N,1) -  matriz (1,N) = matriz (N,N)
        dy = coords[:, None, 1] - coords[None, :, 1]  
        dist = np.sqrt(dx*dx + dy*dy) # distancia
        dist = dist[dist > 0] # eliminamos los cerod de la diagonal (distancia consigo misma)

        h_frame, _ = np.histogram(dist, bins=r_bins)
        rho_frame = N / A_total
        g_frame = h_frame / (N * rho_frame * shell_area) # definicion

        h_total += g_frame

    g_r = h_total / nframes_valid
    r_vals_scaled = r_vals / d_part

    return r_vals_scaled, g_r


# Aceleración Gamma = 4.05
f45 = pd.read_hdf('/data5TB/jpolobar/granular/trayectorias_gas45.h5')
fase45 = f45[(f45['frame'] >= fase1_inicio) & (f45['frame'] <= fase1_fin)]

r45, g45 = calcular_gr(
    fase45,
    frames=range(fase1_inicio, fase1_fin, 100),
    dr=0.01,
    D=25.6,
    d_part=sigma
)

# Aceleración Gamma = 6.30
f70 = pd.read_hdf('/data5TB/jpolobar/granular/trayectorias_gas70.h5')
fase70 = f70[(f70['frame'] >= fase1_inicio) & (f70['frame'] <= fase1_fin)]

r70, g70 = calcular_gr(
    fase70,
    frames=range(fase1_inicio, fase1_fin, 100),
    dr=0.01,
    D=25.6,
    d_part=sigma
)

# GRAFICAR

plt.figure(figsize=(6.4, 4))

plt.plot(r45, g45, marker='o', linestyle='-', markersize=3,markerfacecolor='white', lw=1.5,
         color='#348ABD', label=r'$\Gamma = 4.05$')

plt.plot(r70, g70, marker='s', linestyle='-', markersize=3,markerfacecolor='white', lw=1.5,
         color='#A60628', label=r'$\Gamma = 6.30$')

plt.xlabel(r"$r/\sigma$")
plt.ylabel(r"$g(r)$")
plt.xlim(-0.02, 6)
plt.ylim(-0.02,2.5)
plt.yticks(np.arange(0, 2.51, 0.25))
plt.legend(frameon=False, fontsize=9, loc='best')
plt.grid(False)
plt.tight_layout()
plt.savefig('radial.png', dpi=600, bbox_inches='tight')
plt.show()
