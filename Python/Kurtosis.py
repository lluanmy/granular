################## KURTOSIS DE LA DISTRIBUCIÓN  #########################
# importar librerías 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import cv2
import os
import glob
import gc
import scipy
from scipy.stats import norm,kurtosis
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


frame_rate = 1000  # frames por segundo
start_frame = 6000  # incio vibración aprx
end_frame = 25000 # fin vibración aprx
cols = ['frame', 'vx','vy']

# leer datos
df1 = pd.read_pickle('/data/jpolobar/granular/filtrado_45.pkl')
gas1 = df1.loc[(df1['frame'] >= 6000) & (df1['frame'] <= end_frame), cols]

df2 = pd.read_pickle('/data/jpolobar/granular/filtrado_50.pkl')
gas2 = df2.loc[(df2['frame'] >= 6000) & (df2['frame'] <= end_frame), cols]

df3 = pd.read_pickle('/data/jpolobar/granular/filtrado_60.pkl')
gas3 = df3.loc[(df3['frame'] >= 6000) & (df3['frame'] <= end_frame), cols]

df4 = pd.read_pickle('/data/jpolobar/granular/filtrado_70.pkl')
gas4 = df4.loc[(df4['frame'] >= 6000) & (df4['frame'] <= end_frame), cols]

Gamma = np.array([4.05,4.50,5.40,6.30])
gases = {
    4.05: gas1,
    4.50: gas2,
    5.40: gas3,
    6.30: gas4}

# inicializar
kurt_vals = []
kurt_err  = []

for gamma, df in gases.items():

    df_fase = df[['frame', 'vx', 'vy']].dropna()

    def kurt_frame(group):
        # componente x 
         v = group['vx'].values 

        # Velocidad característica 
        v0 = np.sqrt(np.mean(v**2))

        # Velocidad reducida
        c = v / v0

        # Kurtosis 
        return kurtosis(c, fisher=False)

    # Kurtosis por frame
    k_frame = df_fase.groupby('frame').apply(kurt_frame, include_groups= False)

    # Promedio temporal
    kurt_vals.append(k_frame.mean())
    kurt_err.append(k_frame.std(ddof=1) / np.sqrt(len(k_frame))) # error estándar de la media


kurt_vals = np.array(kurt_vals)
kurt_err  = np.array(kurt_err)

# graficar
plt.figure(figsize=(6.4,4))

plt.errorbar(
    Gamma, kurt_vals, yerr=kurt_err,
    fmt='o',
    ms=4,
    mfc='white',
    mec='black',
    mew=1.2,
    lw=1.5,
    capsize=1.2,
    capthick=1.2,
    color='#A60628',
    label='Datos experimentales'
)
plt.xlabel(r'$\Gamma$')
plt.ylabel(r'$K$')

plt.xlim(4.04,6.33)
plt.ylim(5.1, 7.51)
plt.yticks(np.arange(5.1, 7.51, 0.20))
plt.xticks(np.arange(4.0, 6.31, 0.23))
plt.legend(frameon=False, fontsize=9, loc='upper right')

plt.grid(False)
plt.tight_layout()
plt.savefig('kurtosis.png', dpi=600, bbox_inches='tight')
plt.show()
