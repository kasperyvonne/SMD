import numpy as np
import matplotlib.pyplot as plt

# Diese Tabelle habe ich abgetippt, hoffentulich ohne Fehler :D
Temperatur = np.array([29.4, 26.7, 28.3, 21.1, 20.0, 18.3, 17.8, 22.2, 20.6,
                       23.9, 23.9, 22.2, 27.2, 21.7])
Vorhersage = np.array([2, 2, 1, 0, 0, 0, 1, 2, 2, 0, 2, 1, 1, 0])
Feuchtigkeit = np.array([85, 90, 78, 96, 80, 70, 65, 95, 70, 80, 70,
                        90, 75, 80])
Wind = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1])
Fussball = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

# Definitionen und bums
def Entropie(zp, zn, z):
    wertP = zp / z
    wertN = zn / z
    if (wertP != 0 and wertN != 0):
        return(-wertP*np.log2(wertP)-wertN*np.log2(wertN))
    if (wertP == 0 and wertN != 0):
        return(-wertN*np.log2(wertN))
    if (wertP != 0 and wertN == 0):
        return(-wertP*np.log2(wertP))


def IG(testarray, zielarray, cut, Hz):
    nTruePos = 0
    nTrueNeg = 0
    nFalsePos = 0
    nFalseNeg = 0
    lenPos = len(testarray[testarray > cut])
    lenNeg = len(testarray[testarray < cut])
    lenGes = len(zielarray)
    for i in np.arange(len(testarray)):
        if (testarray[i] > cut and zielarray[i] == 1):
            nTruePos += 1
        if (testarray[i] > cut and zielarray[i] == 0):
            nTrueNeg += 1
        if (testarray[i] < cut and zielarray[i] == 1):
            nFalseNeg += 1
        if (testarray[i] < cut and zielarray[i] == 0):
            nFalsePos += 1
    Htrue = Entropie(nTruePos, nTrueNeg, lenPos)
    Hfalse = Entropie(nFalsePos, nFalseNeg, lenNeg)
    Gain = Hz-((lenPos/lenGes)*Htrue + (lenNeg/lenGes)*Hfalse)
    return(Gain)


# Entropie der Wurzel des Baumes
HBaum = Entropie(len(Fussball[Fussball == 1]), len(Fussball[Fussball == 0]),
                 len(Fussball))

# gewählte Cuts, so dass immer Werte drüber und drunter liegen
Temperaturcuts = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
Vorhersagecuts = [0.5, 1.5]
Feuchtigkeitcus = [70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90]

# Informationsgewinn für die cuts testen
GainT = []
GainV = []
GainF = []
for c in Temperaturcuts:
    GainT.append(IG(Temperatur, Fussball, c, HBaum))
for d in Vorhersagecuts:
    GainV.append(IG(Vorhersage, Fussball, d, HBaum))
for e in Feuchtigkeitcus:
    GainF.append(IG(Feuchtigkeit, Fussball, e, HBaum))

# größten Informationsgewinn finden

bestT = np.argsort(GainT)[-1]
bestV = np.argsort(GainV)[-1]
bestF = np.argsort(GainF)[-1]
print(GainT[bestT])
print(GainV[bestV])
print(GainF[bestF])

#  Plotten des Informationsgewinns

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(Temperaturcuts, GainT, 'rx')
ax1.set_ylabel('Temperatur')
ax2.plot(Vorhersagecuts, GainV, 'bx')
ax2.set_ylabel('Vorhersage')
ax3.plot(Feuchtigkeitcus, GainF, 'gx')
ax3.set_ylabel('Feuchtigkeit')
plt.savefig('Informationsgewinn.pdf')
