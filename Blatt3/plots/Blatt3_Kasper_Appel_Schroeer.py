def Aufgabe1():

    # Definitionen und so
    def Planck(x):
        return(15/(np.pi)**4)*(x**3)/(np.exp(x)-1)

    def PlanckAbl(x):
            return((15/(np.pi)**4)
                   * ((3*(x**2)/(np.exp(x)-1))
                   - (((x**3)*np.exp(x))/(np.exp(x)-1)**2)))

    def SchnittMajo(x, ymax):
        return (ymax-200*(15/(np.pi)**4) * (x**(-0.1)) * (np.exp(-x**(0.9))))

    def Majorante2(x):
        return(200*(15/(np.pi)**4) * (x**(-0.1)) * (np.exp(-x**(0.9))))

    def Majorante1(x):
        return(0.21888647009110665)

    # Vorbereitungsbums
    x = np.linspace(0.0001, 20, 10000)
    xmax = optimize.brentq(PlanckAbl, 2.5, 5)
    ymax = Planck(xmax)
    xs = optimize.brentq(SchnittMajo, 4.5, 6, args=(ymax))
    plt.plot(x, Majorante2(x))
    plt.plot(x, Planck(x))
    plt.ylim(0, 0.3)
    plt.axhline(y=ymax)
    plt.axvline(x=xs, linestyle=':')
    plt.savefig('Majoranten.png')
    plt.clf()
    # Aufgabenteil a)

    def Rueckweisung():
        zaehler = 0
        Verworfenezahlen = 0
        Zufallszahlen = np.empty((100000, 2))
        while zaehler < 100000:
            xwert = np.random.uniform(0, 20)
            ywert = np.random.uniform(0, ymax)
            if(ywert <= Planck(xwert)):
                Zufallszahlen[zaehler] = [xwert, ywert]
                zaehler += 1
            else:
                Verworfenezahlen += 1
        print("Es werden", Verworfenezahlen, "Zahlen verworfen")
        return(Zufallszahlen)

    # Zeit = timeit.timeit(TEST_RUECKWEISUNG, number=1)
    start = time.time()
    Zahlen = Rueckweisung()
    end = time.time()
    print("ymax =", ymax)
    print("Der Schnittpunkt liegt bei x=", xs)
    print("Die Rückweisungsmethode aus a) braucht", (end-start), "Sekunden")
    plt.hist(Zahlen[:, 0], bins=50)
    plt.savefig('Histogramm.png')


def Aufgabe2():

    ## Dat Algorithm ##
    def Metropolis(xzero,num, step,PDF):
    	position = xzero
    	countDooku = 0 # d)
    	randoms = np.array([])
    	Iteration = np.array([]) # d)
    	while (num != randoms.size):
    		xnext = np.random.uniform(position-step,position + step,1)
    		countDooku+=1 # d)
    		P =  min(1,PDF(xnext,xnext-step,xnext+step) / PDF(position,position-step,position+step))
    		rand = np.random.uniform(0,1,1)
    		if rand <= P:
    			position = xnext
    			randoms = np.append(randoms,position)
    			Iteration = np.append(Iteration,countDooku) # d)

    		else:
    			continue
    	return randoms,Iteration # a), d)
    ## Whole lotta Plancking ##
    #def Planckdestr(left,right,num):
    #	if left <0:
    #		return stat.planck.rvs(lambda_,size=num)
    #	else:
    #		return stat.planck.rvs(lambda_,loc=left,size=num)
    #
    #def PlanckPDF(x,left,right):
    #	if left <0:
    #		return stat.planck.pmf(x,lambda_)
    #	else:
    #		return stat.planck.pmf(x,lambda_,loc = left)
    def PlanckPDFBlatt(x,left,right):
    	if left <0:
    		return 0
    	else:
    		return (15/np.pi**4)*(x**3)/(np.exp(x) -1)


    ## Planck Plott ##
    plancks,planckitis = Metropolis(30,10**5,2,PlanckPDFBlatt)
    plt.hist(plancks,bins= 'auto',density = 'True',histtype='step', label="Metropolis-Daten")
    x = np.linspace(min(plancks),max(plancks),1000)
    plt.plot(x,PlanckPDFBlatt(x,min(plancks),max(plancks)),label= "Planck-Verteilung")
    plt.legend(loc='best')
    plt.xlabel(r'Zufallsvariablen')
    plt.ylabel(r'Wahrscheinlichkeit')
    plt.grid()
    plt.savefig('Planckvergleich.pdf')
    plt.clf()
    ## Trace Plot ##
    plt.title('Trace Plot')
    plt.plot(planckitis,plancks,linewidth = 0.001)
    plt.xlabel(r'Iterationen')
    plt.ylabel(r'Zufallsvariablen')
    plt.grid()
    #plt.legend(loc='best')
    plt.savefig('Traceplot.pdf')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import optimize
    import time

    Aufgabe1()
    Aufgabe2()
