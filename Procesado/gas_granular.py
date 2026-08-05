# librerías necesarias
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

# leer película
frames = pims.Cine('/data5TB/jpolobar/granular/gas1.cine')
'''
print(len(frames)) 
print(frames[0])
print(frames[0].shape)
print("dtype:", frames[0].dtype)
'''

#plt.imshow(frames[0],cmap='gray')
#plt.show()

# búsqueda de características en la imagen estática
f = tp.locate(frames[0],7,separation=1.5,minmass=200) # se aconseja un numero impar grande (tras prueba y error he optado por el tamaño de 7)
#print(fm.head())
'''
figure, ax = plt.subplots()

tp.annotate(f,frames[0],ax=ax)  # visualización de las partículas detectadas 
plt.show()


tp.subpx_bias(f) # observamos que NO aparecen caidas en el medio (precision subpixel)
plt.show()
'''
'''
fig, ax = plt.subplots()

ax.hist(f['mass'],bins=20)
ax.set(xlabel='masa', ylabel='cuenta')

fig.show()
'''
#f_mean = f.groupby('frame').mean()
'''
# Crear figura con dos subplots (brillo y excentricidad)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 4))
plt.subplots_adjust(wspace=0.3)

# Dibujar directamente en cada eje
tp.mass_size(f, ax=axs[0])
axs[0].set_xlabel('Brillo total [u.a.]',labelpad=10)
axs[0].set_ylabel('tamaño [px]',labelpad=10)
axs[0].grid(False)

tp.mass_ecc(f, ax=axs[1])
axs[1].set_xlabel('Brillo total [u.a.]',labelpad=10)
axs[1].set_ylabel('excentricidad',labelpad=10)
axs[1].grid(False)

plt.tight_layout()
#plt.savefig('img/tamaño_masa.png', dpi=600, bbox_inches='tight')
plt.show()
'''
'''
with tp.PandasHDFStore('gas50.h5') as store:
    tp.batch(frames, diameter=7, minmass=200, separation=1.5, output=store) # cada uno de los fotogramas del video

'''
'''
store = pd.HDFStore('gas50.h5')
frames = []

for key in store.keys():
    df = store[key]
    df['frame'] = int(key.split('_')[-1])  # extrae el número de frame
    frames.append(df)

store.close()

f = pd.concat(frames, ignore_index=True)
f.to_hdf('caracteristicas_gas50.h5', key='features', mode='w')

'''
'''
print(" Leyendo características...")
p = pd.read_hdf('caracteristicas_gas50.h5', key='features')


print(f"DataFrame cargado: {len(p)} filas, {p['frame'].nunique()} frames")


print(" Enlazando trayectorias... ")
t = tp.link_df(p, search_range=6, memory=4) # trayectorias 
del p
gc.collect()
print("completado")

store = pd.HDFStore('trayectorias_gas50.h5', mode='w') # guardar trayectorias por bloque (RAM)
bloque = 1000000  

n = len(t)
for start in range(0, n,bloque):
    end = min(start + bloque, n)
    print(f" Guardando filas {start}-{end}")
    store.append('linked', t.iloc[start:end], format='table', data_columns=True)
    gc.collect()

store.close()
del t
gc.collect()

print(" Guardado completo")

'''
'''
cols = ['particle', 'frame']
t = pd.read_hdf('/data5TB/jpolobar/granular/trayectorias_gas50.h5', columns=cols)
#print(t.head())
#t = t[((t['mass'] > 200) & (t['size'] < 1.5) & (t['ecc'] < 0.5))]
t1 = tp.filter_stubs(t, 800) # eliminar trayectorias que duran menos de 0.8 s
print('Antes:', t['particle'].nunique())
print('Después:', t1['particle'].nunique())

t1 = t1.reset_index(drop=True)
counts = t1.groupby('frame').size()
print(counts.describe()) # media, desviación, min, max, etc.

Nframes = len(counts)
media = counts.mean()
std = counts.std(ddof=1)
sem = std / np.sqrt(Nframes)
minimum, maximum = counts.min(), counts.max()

print(f"Frames: {Nframes}")
print(f"Media por frame: {media:.4f}")
print(f"Desviación estándar: {std:.4f}")
print(f"Error estándar (sem): {sem:.4f}")
print(f"Min, Max observados: {minimum}, {maximum}")

# Intervalo de confianza 95%
ci_low = media - 1.96 * sem
ci_high = media + 1.96 * sem
print(f"IC 95% para la media: [{ci_low:.2f}, {ci_high:.2f}]")

# Comparación con valor teorico
n_teo = 5234
z = (media - n_teo) / sem

pval = 2 * norm.sf(abs(z))


print(f"Z = {z:.3f}, p-value (two-sided) ≈ {pval:.3e}")

# Porcentaje de discrepancia
rel_diff = (media - n_teo) / n_teo * 100
print(f"Diferencia relativa media vs teórico: {rel_diff:.4f} %")

# evolución detecciones por frame
plt.figure(figsize=(6.4,4))
#plt.subplot(1,2,1)
plt.plot(counts.index/1000, counts.values, '.', markersize=2)
plt.xlim(0,35)
plt.ylim(4900,5720)
plt.axhline(media,lw=1.5, color='red',linestyle='dashed', label=r'$\overline{N} = 5314$')
plt.axhline(n_teo,lw=1.5, color='#467821', linestyle='dotted', label=r'$N_\mathrm{t} =5234 $')
plt.xlabel('t [s]')
plt.ylabel('Número de partículas detectadas')
plt.legend(frameon=False, fontsize=9, loc='upper center')
plt.grid(False)
plt.tight_layout()
plt.savefig('evolucion.png', dpi=600, bbox_inches='tight')
plt.show()
'''
'''
plt.subplot(1,2,2)
plt.hist(counts.values, bins=100)
plt.axvline(media, color='red')
plt.axvline(n_teo, color='green', linestyle='--')
plt.title('Histograma de conteos por frame')
plt.tight_layout()
plt.show()
'''




#######################################################33
'''
# Agrupar por partícula
t_mean = t.groupby('particle').mean()

# Crear figura con dos subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.subplots_adjust(wspace=0.3)

# Dibujar directamente en cada eje
tp.mass_size(t_mean, ax=axs[0])
axs[0].set_xlabel('masa',labelpad=10)
axs[0].set_ylabel('tamaño',labelpad=10)
axs[0].grid(False)

tp.mass_ecc(t_mean, ax=axs[1])
axs[1].set_xlabel('masa',labelpad=10)
axs[1].set_ylabel('excentricidad',labelpad=10)
axs[1].grid(False)

plt.tight_layout()
#plt.savefig('img/masa_tamaño.png', dpi=600, bbox_inches='tight')
plt.show()
'''


#t2 = t[((t['mass'] > 200) & (t['size'] < 1.5) &
#         (t['ecc'] < 0.5))]

'''
figure, ax = plt.subplots()

tp.annotate(t2[t2['frame'] == 16583], frames[16583],ax=ax) # otro frame distinto (vibración)
plt.show()

num_particles = len(t2[t2['frame'] == 0])
print(f"Número de partículas en el frame 0: {num_particles}")
counts = t2.groupby('frame').size()


plt.figure(figsize=(8,4))
counts.plot()
plt.xlabel('Frame')
plt.ylabel('Número de partículas detectadas')
plt.title('Evolución del número de detecciones por frame')
plt.grid(True)
plt.show()


print(counts.mean())
print(counts.max())
print(counts.min())
frame_max = counts.idxmax()   # frame con valor máximo
n_max = counts.max()          # valor máximo

print(f"El frame con más partículas es {frame_max}, con {n_max} partículas detectadas.")
print(f"El frame con menos partículas es {counts.idxmin()}, con {counts.min()} partículas detectadas.")


# Definir frames de vibración
start_frame = 7500   
end_frame = 25000

'''

#######################################VELOCIDADES############################################################################
'''
print(f"DataFrame cargado: {len(t2)} filas, {t2['particle'].nunique()} partículas")


#Calculamos velocidades si no existen

if 'vx' not in f.columns:
    dt = 0.001
    t2 = t2.reset_index(drop=True)
    f = t2.sort_values(['particle', 'frame'])
    
    # Diferencias
    dx= f.groupby('particle')['x'].diff()
    dy= f.groupby('particle')['y'].diff()
    f['vx'] = dx / dt
    f['vy'] = dy / dt
    f['v'] = np.sqrt(f['vx']**2 + f['vy']**2)
    
    # Guardamos progresivamente en HDF5

    store = pd.HDFStore('trayectorias_con_vel_gas50.h5', mode='w')
    bloque = 1_000_000  # 1 millón de filas por bloque
    n = len(f)
    
    for start in range(0, n, bloque):
        end = min(start + bloque, n)
        store.append('features', f.iloc[start:end], format='table', data_columns=True)
        print(f" Guardando filas {start}-{end}")
        gc.collect()  # liberar memoria entre bloques

    store.close()
    del f
    gc.collect()
    print(" Guardado completo en 'trayectorias_con_vel.h5'")

else:
    print(" Las velocidades ya estaban calculadas.")

'''
'''
print("Buscando trayectorias completas...")

# Leemos las 2 columnas solo para que sea rápido
df_counts = pd.read_hdf('/data/jpolobar/granular/trayectorias_gas70.h5', columns=['frame', 'particle'])

total_frames = df_counts['frame'].nunique()
# Contamos cuántos frames tiene cada partícula 
counts = df_counts['particle'].value_counts() 

# Filtramos las que duran todo el video
trayectorias_completas = counts[counts >= 30000].index.values 

print(f"Trayectorias completas encontradas: {len(trayectorias_completas)}")



# leemos el dataset completo pero solo para las partículas elegidas
del(df_counts) 
df = pd.read_hdf('/data/jpolobar/granular/trayectorias_gas70.h5')
df = df[['frame', 'particle', 'x', 'y']].rename(columns={'particle': 'track'})
df = df[df['track'].isin(trayectorias_completas)].copy().reset_index(drop=True)

Ntracks = df['track'].nunique()

####  INDIVIDUAL TRACKS ##############
# build a 't_id' individual track 
def track(t_id, tabla, px, dt, dropit):
    t1 = tabla.loc[tabla.track == t_id].reset_index(drop=dropit)
    vx = px * ( t1.x.values[1:] - t1.x.values[:-1] ) / dt
    vy = px * ( t1.y.values[1:] - t1.y.values[:-1] ) / dt
    t1 = t1.iloc[:-1].copy()  # copia explícita
    t1['vx'] = vx
    t1['vy'] = vy
    return t1

# build individual tracks from all kept tracks
def all_tracks(tabla, tracks_reales, px, dt, dropit):
    global tr_lengths
    n_reales = len(tracks_reales)
    tr_lengths = np.empty(n_reales, dtype=int)
    tracks = []
    
    for idx, t_id in enumerate(tracks_reales):
        t_obj = track(t_id, tabla, px, dt, dropit)
        tracks.append(t_obj)
        tr_lengths[idx] = int(len(t_obj))
        
        if idx % 100 == 0: # Imprime cada 100 partículas para ver el progreso
            print(f"Iteración: {idx} (Procesando partícula ID: {t_id})")
            
    return tracks


print("Cargando dataset...")
df = pd.read_hdf('/data/jpolobar/granular/trayectorias_gas70.h5')
df = df[['frame', 'particle', 'x', 'y']].rename(columns={'particle': 'track'})
df = df.reset_index(drop=True)

print("Buscando trayectorias estables...")
# Contamos cuántos fotogramas dura cada partícula
counts = df['track'].value_counts()

# Seleccionamos 
tracks_seleccionados = counts[counts >= 20000].index.values
print(f"Trayectorias estables encontradas: {len(tracks_seleccionados)}")

# Filtramos el DataFrame original 
df = df[df['track'].isin(tracks_seleccionados)].copy().reset_index(drop=True)
Ntracks = df['track'].nunique()

print('No. of tracks estables: ', Ntracks, '\n')


dt = 1 / 1000
px = 1 / ( 704 * 2.3825 / 256 )

tracks = all_tracks(df, tracks_seleccionados, px, dt, True)
tracks_df = pd.concat(tracks, ignore_index=True)
del(df) # Liberamos memoria 

tracks_df = tracks_df.sort_values(by=['frame','track']).reset_index(drop=True)
print(tracks_df.head())

#  velocidad al cuadrado
tracks_df['v2'] = tracks_df['vx']**2 + tracks_df['vy']**2

# Temperatura granular promedio por frame
T_g = 0.5 * tracks_df.groupby('frame')['v2'].mean()

# Filtro gaussiano para suavizar la señal de la gráfica
T_g_filtered = gaussian_filter1d(T_g, sigma=100)


plt.figure(figsize=(10, 5))
plt.plot(T_g.index / 1000, T_g_filtered, ',')
plt.xlabel('Tiempo (s)')
plt.ylabel('T_g')
plt.grid(True)
plt.show()

tracks_df.to_pickle('/data/jpolobar/granular/filtrado_70.pkl')
'''
