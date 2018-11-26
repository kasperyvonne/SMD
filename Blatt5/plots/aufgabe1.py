import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def Signal(x):
    return((1-x)**(-(1.0/1.7)))


def Neutrinofluss(E):
    return(1.7*(E**(-2.7)))


def Akzeptanz(E):
    return((1-np.exp(-0.5*E))**3)


def HitsSim(E):
    i = 0
    x = -1
    while i == 0:
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        v1 = 2*u1-1
        v2 = 2*u2-1
        s = v1**2+v2**2
        if (s < 1):
            term = np.sqrt(-(2/s)*np.log(s))
            c1 = v1*term
            c2 = v2*term
            x = 2*E*c1+10*E
            x = np.round(x)
            if (x > 0):
                i = 1
    return(x)


def OrtsSim(N):
    Treffer = 0
    while Treffer == 0:
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        v1 = 2*u1-1
        v2 = 2*u2-1
        s = v1**2+v2**2
        if (s < 1):
            term = np.sqrt(-(2/s)*np.log(s))
            sig = 1/(np.log10(N+1))
            c1 = v1*term
            c2 = v2*term
            x = sig*c1+sig*c2+7
            y = sig*c2+3
            if (0 <= x <= 10) and (0 <= y <= 10):
                Treffer = 1
    return((x, y))


def OrtBKG(N):
    Treffer = 0
    while Treffer == 0:
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        v1 = 2*u1-1
        v2 = 2*u2-1
        s = v1**2+v2**2
        if (s < 1):
            term = np.sqrt(-(2/s)*np.log(s))
            sig = 3
            c1 = v1*term
            c2 = v2*term
            x = np.sqrt(0.75)*sig*c1+0.5*sig*c2+5
            y = 0.5*sig*c2+5
            if (0 <= x <= 10) and (0 <= y <= 10):
                Treffer = 1
    return((x, y))


# Aufgabenteil a)
Esize = 100000
x = np.random.uniform(0, 1, Esize)
Energie = np.empty(Esize)
Energie = Signal(x)

df = pd.DataFrame()
# Ein DataFrame für die Energiemessung und für die AcceptanceMask da diese gleich lang sind
df['Energy'] = Energie[0:]
# Aufgabenteil b)
x2 = np.random.uniform(0, 1, Esize)
Detektiert = np.zeros(Esize, dtype=bool)

for i in np.arange(0, Esize):
    if x2[i] < Akzeptanz(Energie[i]):
        Detektiert[i] = True

DetEnergie = Energie[Detektiert]
plt.hist(Energie, bins=np.logspace(0, 3), histtype='step',
         label='Energie')
plt.hist(DetEnergie, bins=np.logspace(0, 3), histtype='step', linestyle='--',
         label='Detektierte Energie')
plt.loglog()
plt.legend()
plt.xlabel(r'$E/\mathrm{TeV}$')
plt.ylabel(r'Ereignisse')
plt.savefig('Energie.pdf')
plt.clf()
print(len(DetEnergie))
df['AcceptanceMask'] = Detektiert[0:]
# Aufgabenteil c)
Hits = np.zeros(len(DetEnergie), dtype=int)
for j in np.arange(0, len(DetEnergie)):
    Hits[j] = HitsSim(DetEnergie[j])
plt.hist(Hits, bins=50, range=[0, 100], histtype='step')
plt.xlabel(r'$Anzahl der Hits$')
plt.savefig('Hits.pdf')
plt.clf()
df2 = pd.DataFrame()
# Ein weiters DataFrame da die weiteren Daten nur auf den akzeptierten Energien
# berechnet werden.
df2['NumberofHits'] = Hits[0:]
# Aufgabenteil d)

Ort = np.zeros((len(DetEnergie), 2))
for k in np.arange(0, len(DetEnergie)):
    Ort[k:] = OrtsSim(Hits[k])

plt.hist2d(Ort[:, 0], Ort[:, 1], bins=[50, 50],
           range=[[0, 10], [0, 10]], cmap='viridis')
plt.plot(7, 3, 'rx')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.savefig('Ort.pdf')
plt.clf()
df2['x'] = Ort[:, 0]
df2['y'] = Ort[:, 1]

# Aufgabenteil e)

BKGSize = 100000
mu = 2
sigma = 1
log10_N_BKG = np.random.normal(mu, sigma, BKGSize)
N_BKG = 10**log10_N_BKG
Ort_BKG = np.zeros((BKGSize, 2))

for l in tqdm(np.arange(0, BKGSize)):
    Ort_BKG[l:] = OrtBKG(N_BKG[l])

df_BKG = pd.DataFrame()
df_BKG['NumberofHits'] = N_BKG[0:]
df_BKG['x'] = Ort_BKG[:, 0]
df_BKG['y'] = Ort_BKG[:, 1]

plt.hist2d(Ort_BKG[:, 0], Ort_BKG[:, 1], bins=[50, 50],
           range=[[0, 10], [0, 10]], cmap='viridis')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.savefig('Ort_BKG.pdf')
plt.clf()

plt.hist(log10_N_BKG, bins=48, range=[0, 6], histtype='step',
         label='Background')
plt.legend()
plt.xlabel(r'$Logarithmus der Anzahl der Hits$')
plt.savefig('Hits_BKG.pdf')
plt.clf()
dfSignal = pd.concat([df, df2], axis=1)
print(dfSignal)
dfSignal.to_hdf('NeutrinoMC.hdf5', key='Energy')
df_BKG.to_hdf('NeutrinoMC.hdf5', key='Background')
