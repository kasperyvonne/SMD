import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


def Planck(x):
    return(15/(np.pi)**4)*(x**3)/(np.exp(x)-1)


def PlanckAbl(x):
        return((15/(np.pi)**4)
               * ((3*(x**2)/(np.exp(x)-1))
               - (((x**3)*np.exp(x))/(np.exp(x)-1)**2)))


def SchnittMajo(x, ymax):
    return (ymax-200*(15/(np.pi)**4) * (x**(-0.1)) * (np.exp(-x**(0.9))))


def Majorante(x):
    # if (x <= xs):
    #     return(ymax)
    #else:
    return(200*(15/(np.pi)**4) * (x**(-0.1)) * (np.exp(-x**(0.9))))


x = np.linspace(0.0001, 20, 10000)
xmax = optimize.brentq(PlanckAbl, 2.5, 5)
ymax = Planck(xmax)
xs = optimize.brentq(SchnittMajo, 4.5, 6, args=(ymax))


print(xs)
plt.plot(x, Majorante(x))
plt.plot(x, Planck(x))
plt.ylim(0,0.3)
plt.axhline(y=ymax)
plt.show()
