import numpy as np
import matplotlib.pyplot as plt


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


n = 81
a = np.arange(0, n)
b = Periodenlaenge(n, 0)
print('Die Periodenlänge ist maximal (also P(a)=m=1024) bei a=')
print(a[b == 1024])
plt.plot(a, b, label='P(a)')
plt.ylim(0, 1200)
plt.xlabel(r'Multiplikator $a$')
plt.ylabel(r'Periodenlänge $P(a)$')
plt.axhline(y=1024, linewidth=1, linestyle='--', color='r')
plt.legend(loc='best')
#plt.savefig('Periodenlaenge2a.pdf')
plt.show()
