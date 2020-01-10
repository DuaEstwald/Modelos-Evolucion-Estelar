# A script to extract all the dates

import glob
import numpy as np
import matplotlib.pyplot as plt



def data(txt):
	f = open(txt,'r')
	return np.loadtxt(f.read().splitlines()[3:], unpack = True)


dat = glob.glob('tablesZ014/*.dat')

dats = [data(d) for d in dat]
files = [f[11:-4] for f in dat]

# Para sacar el header de los datos sacamos el header de cualquiera, escogemos el primero

h = open(dat[0],'r')
header = h.read().splitlines()[0].split()

# Sacamos tambien las unidades por si acaso
h = open(dat[0],'r')
units = h.read().splitlines()[1].split()


# Importamos ahora el fichero inicial del modelo M=1Msol, vrot=0, Z=Zsol
dat0 = data('M001Z14V0.dat.txt')


# ===============================================================================================
# ========================================== PRACTICA ===========================================
# ===============================================================================================


# Ej.1 Para los modelos con M=5y9 con rotacion inicial 0, pintar la variacion de la abundancia central de H y He con el tiempo

M5 = np.array(dats)[[('M005' in name)&('V0' in name) for name in files]]
M9 = np.array(dats)[[('M009' in name)&('V0' in name) for name in files]]


# Para pintar las cosas: 

H_c = [mm[['1H_cen' in h for h in header]] for mm in (M5[0],M9[0])]
He_c = [mm[['4He_cen' in h for h in header]] for mm in (M5[0],M9[0])]
time = [mm[['time' in h for h in header]]*1e-6 for mm in (M5[0],M9[0])]

label = [r'$M = 5M_\odot$', r'$M = 9M_\odot$']

fig = plt.figure()
for m in 0,1:
	fig.add_subplot(1,2,m+1)
	plt.plot(np.log10(time[m][0]),H_c[m][0],label='H central')
	plt.plot(np.log10(time[m][0]),He_c[m][0],label='He central')
	plt.legend()
	plt.ylabel('Abundancias')
	plt.xlabel('log(t)')
	plt.title(label[m])
	plt.tight_layout()
plt.savefig('IMG/fig0.png')


# MODELOS CON LA MASA REQUERIDA EN EL INFORME

Mname = ['M0p8', 'M001', 'M1p25', 'M1p5', 'M002', 'M003', 'M005', 'M009', 'M015', 'M025', 'M040']

Mlabel = [r'0.8 M$_\odot$', r'1 M$_\odot$',r'1.25 M$_\odot$',r'1.5 M$_\odot$',r'2 M$_\odot$',r'3 M$_\odot$',r'5 M$_\odot$',r'9 M$_\odot$',r'15 M$_\odot$',r'25 M$_\odot$',r'40 M$_\odot$',]

V0 = []
V4 = []
for mname in Mname:
    V0.append(np.array(dats)[[(mname in name)&('V0' in name) for name in files]])
    V4.append(np.array(dats)[[(mname in name)&('V4' in name) for name in files]])

# Ej.2 Pinta las trazas evolutivas correspondientes en el diagrama HR.


#v0 = np.array(dats)[['V0' in name for name in files]] # without rotation
#v4 = np.array(dats)[['V4' in name for name in files]] # with rotation

v = [V0, V4]

logTeff = [[dd[m][0][['lg(Teff)' in h for h in header]][0] for m in range(len(dd))] for dd in v]
#logTeff = [[dd[['lg(Teff)' in h for h in header]] for dd in vv] for vv in v] # [v0, v4]
logL = [[dd[m][0][['lg(L)' in h for h in header]][0] for m in range(len(dd))] for dd in v] # [v0, v4]
mass = [[dd[m][0][['mass' in h for h in header]][0] for m in range(len(dd))] for dd in v]
logrhoc = [[dd[m][0][['lg(rhoc)' in h for h in header]][0] for m in range(len(dd))] for dd in v]
logTc = [[dd[m][0][['lg(Tc)' in h for h in header]][0] for m in range(len(dd))] for dd in v]
# Ej.3 Pinta la ZAMS para estrellas en el rango de masas considerado arriba
# Ej.4 Remarca en otro color la evolucion de estas estrellas durante la secuencia principal
# Ej.5 Indica cual es la edad de cada una de estas estrellas cuando alcanza la ZAMS
# Ej.6 Indica cual es la edad de las estrellas cuando sale de la MS

# Hacemos todo esto junto, necesitamos, a parte de lo ya sacado, el tiempo y la abundancia de Hidrogeno


H_c = [[dd[m][0][['1H_cen' in h for h in header]][0] for m in range(len(dd))] for dd in v]
He_c = [[dd[m][0][['4He_cen' in h for h in header]][0] for m in range(len(dd))] for dd in v]
C_c = [[dd[m][0][['12C_cen' in h for h in header]][0] for m in range(len(dd))] for dd in v]
O_c = [[dd[m][0][['16O_cen' in h for h in header]][0] for m in range(len(dd))] for dd in v]
time = [[dd[m][0][['time' in h for h in header]][0]*1e-6 for m in range(len(dd))] for dd in v]




# Para los puntos de la ZAMS, cogemos los puntos iniciales de todo cada array

TZAMS = [[TT[i][0] for i in range(len(logTeff[0]))] for TT in logTeff]
LZAMS = [[tt[i][0] for i in range(len(logL[0]))] for tt in logL] # logL en ZAMS (Ej 7.)

# Calculamos los puntos que pertenecen a la secuencia principal para la logTeff y la logL, time

TMS = [[logTeff[ii][i][H_c[ii][i] != 0] for i in range(len(logTeff[0]))] for ii in range(len(logTeff))]

LMS = [[logL[ii][i][H_c[ii][i] != 0] for i in range(len(logL[0]))] for ii in range(len(logL))]

TTAMS = [[Tt[t][-1] for t in range(len(TMS[0]))] for Tt in TMS]
LTAMS = [[Ll[l][-1] for l in range(len(LMS[0]))] for Ll in LMS]
tTAMS = [[Ll[l][-1] for l in range(len(LMS[0]))] for Ll in LMS]
# La edad de la estrella cuando alcanza la ZAMS sera tMS[0] y la edad cuando sale de la secuencia principal sera tMS[-1]. (Ej 6.)
tMS = [[time[ii][i][H_c[ii][i] != 0] for i in range(len(time[0]))] for ii in range(len(time))]
tTAMS = [[tim[ti][-1] for ti in range(len(tMS[0]))] for tim in tMS]

# Ya tenemos L de la ZAMS, nos queda calcular la masa, pero el facil

tZAMS = [[tt[i][0] for i in range(len(time[0]))] for tt in time]

MZAMS = [[mm[i][0] for i in range(len(mass[0]))] for mm in mass] # Mass en ZAMS (Ej 7.)
rhocZAMS = [[rr[i][0] for i in range(len(logrhoc[0]))] for rr in logrhoc] # (Ej 8.)
TcZAMS = [[tc[i][0] for i in range(len(logTc[0]))] for tc in logTc] # (Ej 9.)


# para la MS, nos pide Tc y rhoc (Ej 10.)

rhocMS = [[logrhoc[ii][i][H_c[ii][i] != 0] for i in range(len(logrhoc[0]))] for ii in range(len(logrhoc))]
TcMS = [[logTc[ii][i][H_c[ii][i] != 0] for i in range(len(logTc[0]))] for ii in range(len(logTc))]



# Todo medianamente hecho menos el final, solo queda ir representando.

# ==============================================================================
# PLOTS
# Ej2, 3, 4, 5, 6
titl = [r'$V_{rot} = 0$',r'$V_{rot} = 0.4V_{crit}$']
fig1 = plt.figure()
for m in range(len(logTeff)):
    fig1.add_subplot(1,2,m+1)
    for k in range(len(logTeff[m])):
        plt.plot(logTeff[m][k],logL[m][k], label = str(Mlabel[k]))
        plt.plot(TMS[m][k],LMS[m][k],'k')
    plt.legend()
    ax = plt.gca()
    plt.plot(TZAMS[m],LZAMS[m],'r-')
    plt.plot(TTAMS[m],LTAMS[m],'r--')
    plt.xlabel(r'$\log(T_{eff})$')
    plt.ylabel(r'$\log(L)$')
    ax.invert_xaxis()
    plt.title(titl[m])
    plt.tight_layout()
plt.savefig('IMG/fig1.png')

# Ej7, 8, 9



from scipy.optimize import curve_fit
def y(cte0,cte1,M):
    logM = np.log10(M)
    return cte0*logM + cte1

fig2 = plt.figure()
for m in range(len(logTeff)):
    fig2.add_subplot(1,2,m+1)
    plt.plot(np.log10(np.array(MZAMS[m])),LZAMS[m])
    p0 = np.polyfit(np.log10(np.array(MZAMS[m])),np.array(LZAMS[m]),1)
#    plt.plot(np.log10(np.array(MZAMS[m])),y(p0[0],p0[1],np.array(MZAMS[m])),'--',label='curva teorica')
    plt.xlabel(r'$\log(M_{ZAMS})$')
    plt.ylabel(r'$\log(L_{ZAMS})$')
#    plt.legend()
    plt.title(titl[m])
    plt.tight_layout()
plt.savefig('IMG/fig2.png')

fig3 = plt.figure()
for m in range(len(logTeff)):
    fig3.add_subplot(1,2,m+1)
    plt.plot(np.log10(np.array(MZAMS[m])),rhocZAMS[m])
    p1 = np.polyfit(np.log10(np.array(MZAMS[m])),np.array(rhocZAMS[m]),1)
#    plt.plot(np.log10(np.array(MZAMS[m])),y(p1[0],p1[1],np.array(MZAMS[m])),'--',label='curva teorica')
    plt.xlabel(r'$\log(M_{ZAMS})$')
    plt.ylabel(r'$\log(\rho_{c ZAMS})$')
#    plt.legend()
    plt.title(titl[m])
    plt.tight_layout()
plt.savefig('IMG/fig3.png')

fig4 = plt.figure()
for m in range(len(logTeff)):
    fig4.add_subplot(1,2,m+1)
    plt.plot(np.log10(np.array(MZAMS[m])),TcZAMS[m])
    p2 = np.polyfit(np.log10(np.array(MZAMS[m])),np.array(TcZAMS[m]),1)
#    plt.plot(np.log10(np.array(MZAMS[m])),y(p2[0],p2[1],np.array(MZAMS[m])),'--',label='curva teorica')
    plt.xlabel(r'$\log(M_{ZAMS})$')
    plt.ylabel(r'$\log(T_{c ZAMS})$')
#    plt.legend()
    plt.title(titl[m])
    plt.tight_layout()
plt.savefig('IMG/fig4.png')



# PARA LOS MODELOS CON M = 1[1], 3[5], 9[7], 40[10] Msol
# Ej 10.


fig5 = plt.figure()
for m in range(len(logTeff)):
    fig5.add_subplot(1,2,m+1)
    for k in (1,5,7,10):
        plt.plot(logrhoc[m][k],logTc[m][k], label = str(Mlabel[k]))
        plt.plot(rhocMS[m][k],TcMS[m][k],'k')
    plt.xlabel(r'$\log (\rho_{c})$')
    plt.ylabel(r'$\log (T_{c})$')
    plt.legend()
    plt.title(titl[m])
    plt.tight_layout()
plt.savefig('IMG/fig5.png')

# Para el modelo con M = 5[6], V = 0

# Calculamos el tiempo en el que se empieza/termina a quemar el C, He, H.

t_He0 = time[0][6][np.where(He_c[0][6] == np.max(He_c[0][6]))[0][-1]+25] #[EMPIEZA A QUEMAR] Para saber que numero poner en el mask simplemente mira el array del carbono y veras que nunca llega a 0 sino que empieza por 1e-5 hasta que empieza a subir
t_He = time[0][6][He_c[0][6] > 1e-4][-1] #[TERMINA DE QUEMAR]
t_H = time[0][6][H_c[0][6] > 1e-4][-1] #[TERMINA DE QUEMAR]
# Con el oxigeno da un poco igual


fig6 = plt.figure()
fig6.add_subplot(2,2,1)
plt.plot(logTeff[0][6],logL[0][6])
plt.plot(logTeff[0][6][H_c[0][6] > 1e-4][-1],logL[0][6][H_c[0][6] > 1e-4][-1],'ro',label=r'Final de la quema de $H_c$')
plt.plot(logTeff[0][6][He_c[0][6] > 1e-4][-1],logL[0][6][He_c[0][6] > 1e-4][-1],'go',label=r'Final de la quema de $He_c$')
plt.plot(logTeff[0][6][np.where(He_c[0][6] == np.max(He_c[0][6]))[0][-1]+25],logL[0][6][np.where(He_c[0][6] == np.max(He_c[0][6]))[0][-1]+25],'bo',label=r'Inicio de la quema de $He_c$')
plt.plot(TZAMS[0],LZAMS[0],'m')
plt.ylim(2.5,3.6)
plt.xlim(3.5,4.5)
ax = plt.gca()
ax.invert_xaxis()
plt.ylabel(r'$\log(L/L_{\odot})$')
plt.xlabel(r'$\log(T_{eff})$ [K]')
fig6.add_subplot(2,2,2)
plt.plot(time[0][6],H_c[0][6],label=r'$H_c$')
plt.plot(time[0][6],He_c[0][6],label=r'$He_c$')
plt.plot(time[0][6],C_c[0][6],label=r'$C_c$')
plt.plot(time[0][6],O_c[0][6],label=r'$O_c$')
plt.xlim(75,time[0][6][-1])
plt.axvline(x = t_He, color = 'k', ls='--')
plt.axvline(x = t_He0, color = 'k', ls='--')
plt.axvline(x = t_H, color = 'k', ls='--')
plt.ylabel(r'Cantidad de He C y O')
plt.xlabel(r't [Myr]')
plt.legend()
fig6.add_subplot(2,2,3)
plt.plot(logrhoc[0][6],logTc[0][6])
plt.plot(logrhoc[0][6][H_c[0][6] > 1e-4][-1],logTc[0][6][H_c[0][6] > 1e-4][-1],'ro',label=r'Final de la quema de $H_c$')
plt.plot(logrhoc[0][6][He_c[0][6] > 1e-4][-1],logTc[0][6][He_c[0][6] > 1e-4][-1],'go',label=r'Final de la quema de $He_c$')
plt.plot(logrhoc[0][6][np.where(He_c[0][6] == np.max(He_c[0][6]))[0][-1]+25],logTc[0][6][np.where(He_c[0][6] == np.max(He_c[0][6]))[0][-1]+25],'bo',label=r'Inicio de la quema de $He_c$')
plt.plot([0.0,4.25],[7.7,9.0],ls='--', color = 'k')
plt.plot([3.80,7.5],[6.9,9.0],ls='--', color = 'k')
plt.plot([6.3,6.3],[6.9,8.3],ls='--', color = 'k')
plt.ylim(6.9,9.0)
plt.xlim(left=0.0)
plt.ylabel(r'$\log(T_c)$')
plt.xlabel(r'$\log(\rho_c)$ $[g/cm^{3}]$')

# Para calcular el Radio solo tenemos que

def radio(lgL,lgTeff):
    sigma = 5.670373e-8 #[W/m2K4]
    Lsol = 3.827e26 #W
    Rsol = 6.957e8 #m
    return np.sqrt((10**(lgL))*Lsol/(4*np.pi*sigma*(10**(lgTeff))**4))/Rsol



fig6.add_subplot(2,2,4)
plt.plot(time[0][6],radio(logL[0][6],logTeff[0][6]))
plt.plot(time[0][6][H_c[0][6] > 1e-4][-1],radio(logL[0][6][H_c[0][6] > 1e-4][-1],logTeff[0][6][H_c[0][6] > 1e-4][-1]),'ro',label=r'Final de la quema de $H_c$')
plt.plot(time[0][6][He_c[0][6] > 1e-4][-1],radio(logL[0][6][He_c[0][6] > 1e-4][-1],logTeff[0][6][He_c[0][6] > 1e-4][-1]),'go',label=r'Final de la quema de $He_c$')
plt.plot(time[0][6][np.where(He_c[0][6] == np.max(He_c[0][6]))[0][-1]+25],radio(logL[0][6][np.where(He_c[0][6] == np.max(He_c[0][6]))[0][-1]+25],logTeff[0][6][np.where(He_c[0][6] == np.max(He_c[0][6]))[0][-1]+25]),'bo',label=r'Inicio de la quema de $He_c$')
plt.xlim(75,time[0][6][-1])
plt.axvline(x = t_He, color = 'k', ls='--')
plt.axvline(x = t_He0, color = 'k', ls='--' )
plt.axvline(x = t_H, color = 'k', ls='--')
plt.ylabel(r'$R[R_{\odot}]$')
plt.xlabel(r't [Myr]')
plt.tight_layout()
plt.savefig('IMG/fig6.png')



