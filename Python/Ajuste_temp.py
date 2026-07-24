################ REGRESIÓN LINEAL TEMPERATURA  ########################################
# cargar librerías necesarias 
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



frame_rate = 1000  # frames por segundo
start_frame = 6000 # inicio vibración aprox
end_frame = 26000 # fin vibración aprox

cols = ["frame", "vx", "vy", "v2"]

# leer datos de cada conf de gas granular
df1 = pd.read_pickle("/data/jpolobar/granular/filtrado_45.pkl")
gas1 = df1.loc[(df1["frame"] >= start_frame) & (df1["frame"] <= end_frame), cols]

df2 = pd.read_pickle("/data/jpolobar/granular/filtrado_50.pkl")
gas2 = df2.loc[(df2["frame"] >= start_frame) & (df2["frame"] <= end_frame), cols]

df3 = pd.read_pickle("/data/jpolobar/granular/filtrado_60.pkl")
gas3 = df3.loc[(df3["frame"] >= start_frame) & (df3["frame"] <= end_frame), cols]

df4 = pd.read_pickle("/data/jpolobar/granular/filtrado_70.pkl")
gas4 = df4.loc[(df4["frame"] >= start_frame) & (df4["frame"] <= end_frame), cols]

Gamma = np.array([4.05, 4.50, 5.40, 6.30])
gases = {4.05: gas1, 4.50: gas2, 5.40: gas3, 6.30: gas4}

# inicializar
T_mean = []
T_err = []

for gamma, df in gases.items():

    T_g = 0.5 * df.groupby("frame")["v2"].mean() # cálculo de T_g

    # promedio de cada conf
    T_mean.append(T_g.mean())

    # error estándar de la media
    T_err.append(T_mean.std(ddof=1) / np.sqrt(len(T_mean)))

T_mean = np.array(T_mean)
T_err = np.array(T_err)

# ajuste lineal
res = linregress(Gamma, T_mean)
m = res.slope  
b = res.intercept  
r2 = res.rvalue**2  

x_fit = np.linspace(4.01, 6.5, 100)
y_fit = m * x_fit + b

# graficar
plt.figure(figsize=(6.4, 4))
plt.plot(x_fit, y_fit, color='#467821' , linestyle="--", lw=1.1, label='Regresión lineal')

texto = (r'$T_g = {:.4f} \, \Gamma {:.4f}$'
         '\n\n'
         r'$R^2 = {:.4f}$'.format(m,b,r2))

plt.text(4.5, 0.035, texto, fontsize=9, bbox=dict(facecolor='white', edgecolor='grey', boxstyle='square,pad=0.5'))

plt.errorbar(
    Gamma,
    T_mean,
    yerr=T_err,
    fmt="o",
    ms=4,
    mfc="white",
    mec="black",
    mew=1.2,
    lw=1.5,
    capsize=1.2,
    capthick=1.2,
    color="#A60628",
    label='Datos experimentales'
)

plt.xlabel(r"$\Gamma$")
plt.ylabel(r"$T_g\,[\mathrm{\sigma}^2/\mathrm{s}^2]$")
plt.xticks(Gamma)
plt.xlim(4, 6.33)
plt.ylim(0.015,0.040)
plt.legend(frameon=False, fontsize=9, loc='lower right')
plt.grid(False)
plt.tight_layout()
plt.savefig('temp.png', dpi=600, bbox_inches='tight')
plt.show()
