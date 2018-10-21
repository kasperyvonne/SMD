import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const


def funktion1(x):
    y = ((x**3)+(1/3))-((x**3)-(1/3))
    return(y)


def funktion2(x):
    y = ((3+((x**3)/3))-(3-((x**3)/3)))*(x**3)
    return(y)


Werte1 = np.logspace(4, 7, 10000)
Werte2 = np.logspace(0, -4, 10000)

FunkWert1 = funktion1(Werte1)
FunkWert2 = funktion2(Werte2)

Abweichung1 = 2/3-FunkWert1
Abweichung2 = 2/3-FunkWert2
print(FunkWert2)
plt.plot(Werte1, Abweichung1, linewidth=1)
plt.axhline(y=(2/300), linewidth=0.5, color='k', alpha=0.6)
plt.axhline(y=-(2/300), linewidth=0.5, color='k', alpha=0.6)
plt.axhline(y=(2/3), linewidth=0.5, color='r')
plt.xscale('log')
plt.show()
