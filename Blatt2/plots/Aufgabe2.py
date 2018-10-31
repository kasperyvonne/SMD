import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Für a) und e)
# Maximale Periodenlaenge ist m, durch den Modulo Operator,
# suche nach dem wiederauftauchen des seeds x, da dann die Folge von vorne
# beginnt. Da nur die Periodenlänge interessiert, werden die erzeugten
# Zufallszahlen nicht durch m geteilt um eine Verteilung zwischen 0 und 1 zu
# erhalten.
# n ist der Bereich auf dem P untersucht werden soll und x der Seed
def Periodenlaenge(n, x):
    Periodenlaenge = np.empty(n)
    for a in np.arange(0, n):
        Werte = np.empty(1025)
        Werte[0] = x  # Seed festlegen
        flag = False  # bool zur überprüfung ob Seed schon wieder da war
        for i in np.arange(1, 1025):  # über 1024 weil das die max. PL ist
            Werte[i] = ((a*(Werte[i-1])+3) % 1024)
            if((Werte[i] == x) and (flag is False)):
                Periodenlaenge[a] = i  # Index d ersten Wiederholung des Seeds
                flag = True
    return(Periodenlaenge)


# Erzeugt n Zufallszahlen mit einem linear kongruenten Zufallszahlengenerator,
# gibt ein n-komponentiges Array zurück. Dabei ist der Seed x und a,b,m fest
def LinKongruent(x, n):
    Werte = np.empty(n)
    Werte[0] = x
    for i in np.arange(1, n):  # geht die Indices von 1 bis n-1 als integer ab
        Werte[i] = ((1601*(Werte[i-1])+3456) % 10000)
    return(Werte/10000)


# Aufgabenteil a)

n = 81
Wertebereich = np.arange(0, n)
Periodenlaenge = Periodenlaenge(n, 0)
print('Die Periodenlänge ist maximal (also P(a)=m=1024) bei a=')
print(Wertebereich[Periodenlaenge == 1024])
plt.plot(Wertebereich, Periodenlaenge, label='P(a)')
plt.ylim(0, 1200)
plt.xlabel(r'Multiplikator $a$')
plt.ylabel(r'Periodenlänge $P(a)$')
plt.axhline(y=1024, linewidth=1, linestyle='--', color='r')
plt.legend(loc='best')
plt.savefig('Periodenlaenge.png')
plt.clf()

# Aufgabenteil b)

n = 10000
Zufallszahlen = LinKongruent(1, 10000)
plt.hist(Zufallszahlen, bins=99)
plt.xlabel(r'Zufallszahlen')
plt.ylabel(r'Anzahl')
plt.savefig('Zufallszahlen2b.png')
plt.clf()

# Aufgabenteil c)
Zufallszahlen2 = Zufallszahlen.reshape(5000, 2)
x = Zufallszahlen2[:, 0]
y = Zufallszahlen2[:, 1]
plt.scatter(x, y, s=0.3)
plt.savefig('Paare2c.png')
plt.clf()
Zufallszahlen3 = LinKongruent(1, 9999)
Zufallszahlen3 = Zufallszahlen3.reshape(3333, 3)
x = Zufallszahlen3[:, 0]
y = Zufallszahlen3[:, 1]
z = Zufallszahlen3[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.init_view(45, 30) # funktioniert leider nicht
ax.scatter(x, y, z, lw=0, alpha=0.3)
plt.savefig('Triplets2c.png')
plt.clf()

# Aufgabenteil d)
ZZ = np.random.uniform(0, 1, 10000)
ZZ2 = ZZ.reshape(5000, 2)
x = ZZ2[:, 0]
y = ZZ2[:, 1]
plt.scatter(x, y, s=0.3, c='r')
plt.savefig('Paare2d.png')
plt.clf()

ZZ3 = np.random.uniform(0, 1, 9999)
ZZ3 = ZZ3.reshape(3333, 3)
x = ZZ3[:, 0]
y = ZZ3[:, 1]
z = ZZ3[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, lw=0, alpha=0.3, c='r')
plt.savefig('Triplets2d.png')
plt.clf()

# Aufgabenteil e)
