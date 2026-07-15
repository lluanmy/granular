####  INDIVIDUAL TRACKS ##############
# build a 't_id' indivitual track 
def track(t_id, tabla, px, dt, dropit):
    t1 = tabla.loc[tabla.track == t_id].reset_index(drop=dropit)
    vx = px * ( t1.x.values[1:] - t1.x.values[:-1] ) / dt
    vy = px * ( t1.y.values[1:] - t1.y.values[:-1] ) / dt
    t1 = t1.iloc[:-1].copy()  # copia explícita
    t1['vx'] = vx
    t1['vy'] = vy
    return t1

# build individual tracks from all kept tracks
# OUTPUT
# tr_lengths[i]: length of track no.  'i'. The total no. of tracks is stored in 'Ntracks'
def all_tracks(tabla, px, dt, dropit):
    # length of track
    global tr_lengths
    tr_lengths = np.empty(Ntracks,dtype=int)
    tracks = [[] for i in range(Ntracks)]
    for i in range(Ntracks):
        tracks[i] = track(i, tabla, px,  dt , dropit)
        tr_lengths[i] = int(len(tracks[i]))
        if i % 10000 == 0: print(f"Iteración: {i}")
    return tracks




# leemos los datos (Gamma = 6.30)
df = pd.read_hdf('/data5TB/jpolobar/granular/trayectorias_gas70.h5', compression='infer')
print('leido')
df = df[['frame', 'particle', 'x', 'y']]
df = df.rename(columns={'particle': 'track'})
df = df.reset_index(drop=True)
Ntracks = np.max(df.track.values)
print('no. of tracks: ', Ntracks, '\n')
#print(df.head(), "\n")

# Escala temporal y espacial
dt = 1 / 1000
px = 1 / ( 704 * 2.3825 / 256 )

print('Aplicando funciones')
tracks = all_tracks( df, px, dt, True)
tracks_df = pd.concat(tracks, ignore_index=True)
del(df)
#print(tracks[0])

# RE-ORDER to original indexing (by frame and then by particle)
tracks_df.sort_values(by=['frame','track']).reset_index(drop=True)

print(tracks_df.head())


# TEMPERATURA GRANULAR

tracks_df['v2'] = tracks_df['vx']**2 + tracks_df['vy']**2

T_g = 0.5 * tracks_df.groupby('frame')['v2'].mean()
T_g_filtered = savgol_filter(T_g, window_length=53, polyorder=3) #filtro para eliminar ruido
plt.plot(T_g.index/1000,T_g_filtered)


