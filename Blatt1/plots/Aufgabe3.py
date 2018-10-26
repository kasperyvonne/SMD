import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
# from scipy.optimize import curve_fit
# from scipy.optimize import fsolve


def Maxwell(x, m, T):
    y = (4*np.pi*(m/(2*np.pi*const.k*T))*np.exp((-m*(x**2)/(2*const.k*T))*(x**2)))
    return(y)

x = np.linspace(0,100,1000)
print(Maxwell(x,1000,200))
