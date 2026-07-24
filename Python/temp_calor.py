############################### MAPA CALOR T_g ####################################################################
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

# fase colapso inelástico
fase1_inicio = 35000
fase1_fin = 38418
# leer datos
f = pd.read_hdf('/data/jpolobar/granular/filtrado_gas.h5')
fase1 = f[(f['frame'] >= fase1_inicio) & (f['frame'] <= fase1_fin)]

def granular_temp_map(df, Nx=50, Ny=50):

    S = 1 / ( 704 * 2.3825 / 256 ) # escala espacial (sigma/px)

    # Dominio 
    x_min, x_max = 0, df['x'].max()*S
    y_min, y_max = 0, df['y'].max()*S

    x_edges = np.linspace(x_min, x_max, Nx+1)
    y_edges = np.linspace(y_min, y_max, Ny+1)

    heatmaps = []

    for f in df['frame'].unique():
        df_f = df[df['frame'] == f]
        v_limite = df_f['v'].quantile(1) 
        
        
        df_f_filtrado = df_f[df_f['v'] <= v_limite]

        # Histograma ponderado con v^2
        H_v2, _, _ = np.histogram2d(
            df_f_filtrado['x']*S, df_f_filtrado['y']*S,
            bins=[x_edges, y_edges],
            weights=(df_f_filtrado['v']*S)**2
        )

        # Número de partículas en cada celda
        N, _, _ = np.histogram2d(
            df_f_filtrado['x']*S, df_f_filtrado['y']*S,
            bins=[x_edges, y_edges]
        )

        # Temperatura granular por celda
        with np.errstate(divide='ignore', invalid='ignore'):
            T_cell = 0.5 * H_v2 / N

        T_cell = np.nan_to_num(T_cell, nan=0.0)

        heatmaps.append(T_cell)

    # Promedio temporal
    heatmaps_array = np.array(heatmaps)
    T_mean_cells = np.mean(heatmaps_array, axis=0)

    return T_mean_cells, x_min, x_max, y_min, y_max
# llamar funcion
T_mean_cells, x_min, x_max, y_min, y_max = granular_temp_map(fase1, Nx=90, Ny=90)


# Graficar

plt.figure(figsize=(6.4,4))
plt.imshow(T_mean_cells.T, origin='lower',
           extent=[x_min, x_max, y_min, y_max],
           cmap='Spectral_r', aspect='auto',interpolation='bicubic')
plt.colorbar(label=r'$T_g\,[\mathrm{\sigma}^2/\mathrm{s}^2]$')
plt.xlim(0,30)
plt.ylim(0,60)
plt.xlabel(r'$x / \sigma$')
plt.ylabel(r'$y / \sigma$')
plt.grid(False)
plt.gca().invert_yaxis() # Invierte el eje Y 
plt.tight_layout()
plt.savefig('img/calor.png', dpi=600, bbox_inches='tight')
plt.show()
