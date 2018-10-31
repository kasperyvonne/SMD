def Aufgabe1():
    def funktion1(x):
        y = ((x**3)+(1/3))-((x**3)-(1/3))
        return(y)

    def funktion2(x):
        y = ((3+((x**3)/3))-(3-((x**3)/3)))/(x**3)
        return(y)

    Werte1 = np.logspace(4, 7, 10000)
    Werte2 = np.logspace(-4, -7, 10000)

    FunkWert1 = funktion1(Werte1)
    FunkWert2 = funktion2(Werte2)

    Abweichung1 = 2/3-FunkWert1
    Abweichung2 = 2/3-FunkWert2

    print("Die Abweichung für Funktion a) ist größer als 1% ab dem Wert")
    print(Werte1[Abweichung1 > (2/300)][0])
    print("Der Funktionswert für die Funktion a) ist 0 ab dem Wert")
    print(Werte1[Abweichung1 == (2/3)][0])
    print("Die Abweichung für Funktion b) ist größer als 1% ab dem Wert")
    print(Werte2[Abweichung2 > (2/300)][0])
    print("Der Funktionswert für die Funktion b) ist 0 ab dem Wert")
    print(Werte2[Abweichung2 == (2/3)][0])

    plt.plot(Werte1, Abweichung1)
    plt.axhline(y=(2/300), linewidth=0.5, color='k', alpha=0.6)
    plt.axhline(y=-(2/300), linewidth=0.5, color='k', alpha=0.6)
    plt.axhline(y=(2/3), linewidth=0.5, color='r')
    plt.axvline(x=Werte1[Abweichung1 > (2/300)][0], linewidth=0.5, color='k', alpha=0.6 )
    plt.axvline(x=Werte1[Abweichung1 == (2/3)][0], linewidth=0.5, color='r', alpha=0.6 )
    plt.xscale('log')
    plt.xlabel(r'log(x)')
    plt.ylabel(r'2/3-f(x)')
    plt.title('Numerische Abweichung Teil a)')
    plt.savefig('Aufgabe1a.pdf')
    plt.clf()

    plt.plot(Werte2, Abweichung2)
    plt.axhline(y=(2/300), linewidth=0.5, color='k', alpha=0.6)
    plt.axhline(y=-(2/300), linewidth=0.5, color='k', alpha=0.6)
    plt.axhline(y=(2/3), linewidth=0.5, color='r')
    plt.axvline(x=Werte2[Abweichung2 > (2/300)][0], linewidth=0.5, color='k', alpha=0.6 )
    plt.axvline(x=Werte2[Abweichung2 == (2/3)][0], linewidth=0.5, color='r', alpha=0.6 )
    plt.xscale('log')
    plt.xlabel(r'log(x)')
    plt.ylabel(r'2/3-f(x)')
    plt.title('Numerische Abweichung Teil b)')
    plt.savefig('Aufgabe1b.pdf')


def Aufgabe2():

    def stabilGleichung(teta, gamma):
        return (2+np.sin(teta)**2)/(np.sin(teta)**2 + (np.cos(teta)**2)/gamma**2)

    def instabilGleichung(teta, beta):
        return (2+np.sin(teta)**2)/(1-(beta**2)*np.cos(teta)**2)

    def Kondition(teta, beta):
    	return np.absolute(teta*(1-3*(beta**2)*np.sin(teta)* np.cos(teta))/(1-(np.cos(teta)**2)* beta ** 2)*(np.cos(teta)**2 -3))


    # Sexy Data
    tetahalf = np.linspace(0, np.pi, 1000)
    teta = np.linspace(0, 2*np.pi, 1000)
    beta = .99989
    gamma = 0.01022

    # Sexy Plots
    plt.subplot(3, 1, 1)
    plt.plot(teta, stabilGleichung(teta, gamma), label='Stabil')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$')
    plt.legend(loc='best')
    plt.subplot(3, 1, 2)
    plt.plot(teta, instabilGleichung(teta, beta), label='Instabil')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$')
    plt.legend(loc='best')
    plt.subplot(3, 1, 3)
    plt.plot(tetahalf, Kondition(tetahalf, beta), label="Kondition")
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$K$')
    plt.legend(loc='best')
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    plt.savefig('Stabiplot.pdf')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    Aufgabe1()
    Aufgabe2()
