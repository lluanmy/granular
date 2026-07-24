############ AUTOCORRELACIÓN DE LA VELOCIDAD ###########################
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



fase1_inicio = 6000 # inicio de la vibración aprox
fase1_fin = 25000 # fin de la vibración aprox
def compute_vacf_by_frames(df, max_frame=None, origins_step=10):

    # Asegurar orden correcto
    df = df.sort_values(["frame", "track"])

    # Pivot para obtener matrices velocidad: shape (nt, N) nt-- numero de frames   N-- numero de part
    Vx = df.pivot(index="frame", columns="track", values="vx").values   # matriz nt x N
    Vy = df.pivot(index="frame", columns="track", values="vy").values   # Vy[0,2] vel_y particula 2 en el frame 0
    frames = df['frame'].unique()
    nt = len(frames)

    if max_frame is None:
        max_frame = nt - 1

    # Construimos velocidad vectorial:  eje 0-- frame,   eje1-- partícula,  eje2--componente(0,1) = x,y
    V = np.stack([Vx, Vy], axis=2)   # (nt, N, 2) V[0, 2, :] vector (vx,vy) de la partícula 2 en el frame 0

    # Partículas que están en todos los frames 
    mask = ~np.isnan(V).any(axis=(0,2)) # busca si la partícula tiene un NaN en cualquier frame y en cualquir componente
    V = V[:, mask, :]   # Solo partículas sin NaN
    Np = V.shape[1]
  
    # inicializar
    vacfs = []

    # Selección de orígenes t0: 0, 10, 20, 30…
    origins = np.arange(0, nt - max_frame, origins_step)

    for t0 in origins:

        v0 = V[t0]                           # velocidad en t0: (Np, 2)  v0 = V[t0, :, :]
        norm = np.mean(np.sum(v0 * v0, axis=1)) 

        vacf = np.zeros(max_frame + 1)

        for tau in range(max_frame + 1):
            vt = V[t0 + tau]
            prod = np.sum(v0 * vt, axis=1)
            vacf[tau] = np.mean(prod) / norm

        vacfs.append(vacf)

    vacf_prom = np.mean(vacfs, axis=0)
    tiempos = np.arange(max_frame + 1)

    return vacf_prom, tiempos

# leer datos
l = pd.read_pickle('/data/jpolobar/granular/filtrado_45.pkl')
fase1 = l[(l['frame'] >= fase1_inicio) & (l['frame'] <= fase1_fin)]


e = pd.read_pickle('/data/jpolobar/granular/filtrado_60.pkl')
fase2 = e[(e['frame'] >= fase1_inicio) & (e['frame'] <= fase1_fin)]

d= pd.read_pickle('/data/jpolobar/granular/filtrado_70.pkl')
fase3 = d[(d['frame'] >= fase1_inicio) & (d['frame'] <= fase1_fin)]


fase1_recortado = fase1[fase1['frame'] < 8000]
fase2_recortado = fase2[fase2['frame'] < 8000]
fase3_recortado = fase3[fase3['frame'] < 8000]

vacf1, t1 = compute_vacf_by_frames(fase1_recortado,max_frame=50, origins_step=5)
vacf2, t2 = compute_vacf_by_frames(fase2_recortado,max_frame=50, origins_step=5)
vacf3, t3 = compute_vacf_by_frames(fase3_recortado,max_frame=50, origins_step=5) # de 5 en 5 frames
window_len = 7
poly_order = 3

# filtro para eliminar ruido
vacf1_smooth = savgol_filter(vacf1, window_length=window_len, polyorder=poly_order)
vacf2_smooth = savgol_filter(vacf2, window_length=window_len, polyorder=poly_order)
vacf3_smooth = savgol_filter(vacf3, window_length=window_len, polyorder=poly_order)

  
# graficar

fig, ax = plt.subplots(figsize=(6.4, 4))

#  Gráfica principal 
ax.plot(t1/1000, vacf1_smooth, lw=1.5, marker='o', markerfacecolor='white', linestyle='-', markersize=4, label=r'$\Gamma = 4.05$')
ax.plot(t2/1000, vacf2_smooth, lw=1.5, marker='s', markerfacecolor='white', linestyle='-', markersize=4, label=r'$\Gamma = 5.40$')
ax.plot(t3/1000, vacf3_smooth, lw=1.5, color='#467821', marker='^', markerfacecolor='white', linestyle='-', markersize=4, label=r'$\Gamma = 6.30$')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_xlim(-0.0005, 0.0505)
ax.set_ylim(-0.048, 1.016)
ax.set_xlabel(r'$t$ [s]')
ax.set_ylabel(r'$A_v(t)$')
ax.grid(False)
ax.legend(frameon=False, fontsize=9, loc='center', bbox_to_anchor=(0.85, 0.22),) 
# INSET
ax_inset = ax.inset_axes([0.52, 0.52, 0.43, 0.43]) 
vacf1_smooth = savgol_filter(vacf1, window_length=9, polyorder=poly_order)
vacf2_smooth = savgol_filter(vacf2, window_length=9, polyorder=poly_order)
vacf3_smooth = savgol_filter(vacf3, window_length=9, polyorder=poly_order)

ax_inset.plot(t1/1000, vacf1_smooth, lw=1, marker='o', markerfacecolor='white', linestyle='-', markersize=2)
ax_inset.plot(t2/1000, vacf2_smooth, lw=1, marker='s', markerfacecolor='white', linestyle='-', markersize=2)
ax_inset.plot(t3/1000, vacf3_smooth, lw=1, color='#467821', marker='^', markerfacecolor='white', linestyle='-', markersize=2)

ax_inset.axhline(0, color='gray', linestyle='--', linewidth=0.6, alpha=0.7)
ax_inset.set_xlim(0.0095, 0.0352) 
ax_inset.set_ylim(-0.04, 0.019)
ticks_x = np.arange(0.01, 0.036, 0.005) 
ax_inset.set_xticks(ticks_x)
ticks_y = np.arange(-0.04, 0.20, 0.04)
ax_inset.set_yticks(ticks_y)
ax_inset.set_xlabel(r'$t [s]$', fontsize=8, labelpad=2)
ax_inset.set_ylabel(r'$A_v(t)$', fontsize=8, labelpad=2)
ax_inset.tick_params(labelsize=7) # Hacer los números del inset más pequeños para que quepan bien
ax_inset.grid(False)
plt.savefig('autocorrelacion2.png', dpi=600, bbox_inches='tight')
plt.show()


