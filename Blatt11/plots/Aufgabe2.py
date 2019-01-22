import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

def G(x, mu, sigma):
    return ((1/np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - mu)**2)/(2 * sigma**2)))

l_values = [12, 17, 22]
linestyles_values = ['-', '--', ':']

for lamdahh, ls_v in zip(l_values, linestyles_values):
    dist = poisson(lamdahh)
    x = np.arange(-1, 300)

    plt.plot(x, dist.pmf(x), ls=ls_v, color='black',
             label=r'Poisson mit $\lambda=%i$' % lamdahh)
    plt.plot(x, G(x, lamdahh, np.sqrt(lamdahh)),ls=ls_v, color='red', label=r'Gauß mit $\mu= \sigma^2=%i$'%lamdahh )

plt.xlim(-0.5, 40)
plt.ylim(0, 0.2)
plt.legend()
plt.xlabel('$x$')
plt.ylabel(r'$p(x)$')
plt.title('Poisson und Gauß')
plt.savefig('testplot.pdf')
plt.clf()


def Kologomorow(A, B, alpha):
    KulSummeA = np.cumsum(A[0])
    KulSummeB = np.cumsum(B[0])
    Abstand = max(np.abs(KulSummeA - KulSummeB))
    langeA, langeB = len(A[0]), len(A[0])

    d = np.sqrt(langeA * langeB / (langeA + langeB)) * Abstand
    K_alpha = np.sqrt(1/2 * np.log(2 / alpha))

    ablehnen = False
    if(d > K_alpha):
        ablehnen = True
    return ablehnen
#--test- Kologomorow spuckt True aus.
# lamdahh = 8
# mu = lamdahh
# sigma = np.sqrt(lamdahh)
# p = np.random.poisson(lamdahh, 10000)
# g = np.random.normal(mu, sigma, 10000)
# g = np.floor(g)
#
# bins = np.linspace(lamdahh - 5*np.sqrt(lamdahh), lamdahh + 5*np.sqrt(lamdahh), 100)
#
# phist = np.histogram(p, bins=bins, density=True)
# ghist = np.histogram(g, bins=bins, density=True)
# print(Kologomorow(phist,ghist,0.25))

for alpha in (0.05, 0.025, 0.001):
    print('alpha ist:', alpha)
    for lamdahh in range(1, 15, 1):
        mu = lamdahh
        sigma = np.sqrt(lamdahh)
        p = np.random.poisson(lamdahh, 10000)
        g = np.random.normal(mu, sigma, 10000)
        g = np.floor(g)

        bins = np.linspace(lamdahh - 5*np.sqrt(lamdahh), lamdahh + 5*np.sqrt(lamdahh), 100)

        phist = np.histogram(p, bins=bins, density=True)
        ghist = np.histogram(g, bins=bins, density=True)

        Kolo = Kologomorow(phist, ghist, alpha)

        print('bei alpha =', alpha, 'und lambdah=', lamdahh, 'sagt Kolmogorow–Smirnow:', Kolo)

print('Sorry für den Spam')
