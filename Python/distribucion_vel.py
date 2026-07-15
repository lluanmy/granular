### DISTRIBUCION DE VELOCIDADES
fase1_inicio = 6000  # inicio de la vibración aprox
fase1_fin = 25000   # fin de la vibración aprox

# leemos los datos
f= pd.read_pickle('/data5TB/jpolobar/granular/vel45.pkl')
fase1 = f[(f['frame'] >= fase1_inicio) & (f['frame'] <= fase1_fin)]

e = pd.read_pickle('/data5TB/jpolobar/granular/vel60.pkl')
fase2 = e[(e['frame'] >= fase1_inicio) & (e['frame'] <= fase1_fin)]

d= pd.read_pickle('/data5TB/jpolobar/granular/vel70.pkl')
fase3 = d[(d['frame'] >= fase1_inicio) & (d['frame'] <= fase1_fin)]

#una componente de la velocidad (isotropia)
v1 = fase1['vx'].dropna().values 

v2 =fase2['vx'].dropna().values 

v3 =fase3['vx'].dropna().values 

# Calcular velocidad característica v0
v01 = np.sqrt(2*np.mean(v1**2))   #
v02 = np.sqrt(2*np.mean(v2**2))   
v03 = np.sqrt(2*np.mean(v3**2))   


# Escalar
v_scaled1 = v1 / v01
v_scaled2 = v2 / v02
v_scaled3 = v3 / v03


# Histograma conjunto
hist1, bins = np.histogram(v_scaled1, bins=120, range=(-5,5), density=True)
hist2, bins = np.histogram(v_scaled2, bins=120, range=(-5,5), density=True)
hist3, bins = np.histogram(v_scaled3, bins=120, range=(-5,5), density=True)


centers = 0.5 * (bins[1:] + bins[:-1])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Gaussiana
P_gauss = (1/np.sqrt(np.pi)) * np.exp(-centers**2)

plt.figure(figsize=(6.4,4))
#plt.yscale('log')
plt.scatter(centers, hist1,marker='o',facecolor='white',edgecolors='#348ABD', s=20, label=r'$\Gamma = 4.05$')
plt.scatter(centers, hist2,marker='s' ,facecolor='white',edgecolors='#A60628', s=20, label=r'$\Gamma = 5.40$')
plt.scatter(centers, hist3,marker='^' ,facecolor='white',edgecolors='#467821', s=20, label=r'$\Gamma = 6.30$')

plt.plot(centers, P_gauss, color='black', lw=1.5, label='Gaussiana')
plt.xlabel(r'$v/v_0$')
plt.ylabel(r'$P(v/v_0)$')
plt.xlim(-3, 3)
plt.xticks(np.arange(-3, 3.1, 1)) 
plt.yticks(np.arange(0, 0.76, 0.15)) 
plt.ylim(-0.008, 0.75)
plt.legend(frameon=False)
plt.grid(False)
plt.tight_layout()
plt.savefig('dist_vel.png', dpi=600, bbox_inches='tight')
plt.show()
