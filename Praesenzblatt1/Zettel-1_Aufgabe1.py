import numpy as np
import matplotlib.pyplot as plt

groesse, gewicht = np.genfromtxt("Groesse_Gewicht.txt", unpack=True)
#
# fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, ncols=1, sharex=False)
# ax1.hist(gewicht, bins=5)
# ax2.hist(gewicht, bins=10)
# ax3.hist(gewicht, bins=15)
# ax4.hist(gewicht, bins=20)
# ax5.hist(gewicht, bins=30)
# ax6.hist(gewicht, bins=50)
# plt.show()

fig = plt.figure()
sub1 = fig.add_subplot(231)
sub1.hist(groesse, bins=5)

sub2 = fig.add_subplot(232)
sub2.hist(groesse, bins=10)

sub3 = fig.add_subplot(233)
sub3.hist(groesse, bins=15)

sub4 = fig.add_subplot(234)
sub4.hist(groesse, bins=20)

sub5 = fig.add_subplot(235)
sub5.hist(groesse, bins=30)

sub6 = fig.add_subplot(236)
sub6.hist(groesse, bins=50)

plt.tight_layout()
plt.show()
