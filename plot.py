from cProfile import label
from re import X
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import pylab as p

def g(t, A, B):
    true_val = A*sp.jn(2,t) + B*t
    return true_val

data = "fitting.dat"
data_arr = [None]*10
data_arr = np.loadtxt(data, dtype = np.double, usecols = range(10), unpack = True)

sigma = np.logspace(-1,-3,9)

figure_0 = plt.figure(0)
plt.title('Noise and True Value Plot', size = 27)
plt.xlabel('t  ⟶', size = 23)
plt.ylabel('g(t) + n(t)  ⟶', size = 23)
for i in range(1,9):
    plt.plot(data_arr[0], data_arr[i], label = "σ = " + str(round(sigma[i-1], 3)))
plt.plot(data_arr[0], g(data_arr[0], 1.05, -0.105), "--k", label = 'True Value', linewidth = 3)
plt.grid(True)
plt.legend()

figure_1 = plt.figure(1)
plt.title('Errorbar Plot', size = 27)
plt.xlabel('t  ⟶', size = 23)
plt.ylabel('g(t) + n(t)  ⟶', size = 23)
plt.errorbar(data_arr[0][::5], data_arr[1][::5], sigma[0], fmt="ro")
plt.plot(data_arr[0], g(data_arr[0], 1.05, -0.105), "--k", label = 'True Value', linewidth = 3)
plt.grid(True)
plt.legend()
plt.show()

J2 = np.transpose(sp.jn(2,data_arr[0]))
t = np.transpose(data_arr[0])
M = np.c_[J2,t]
p = [1.05, -0.105]

X = np.dot(M,p)

if(X.all() == g(data_arr[0], 1.05,-0.105).all()):
    print("Dot product of Matrix M and p is equal to g(t,A,B)")
else:
    print("Dot product of Matrix M and p is NOT equal to g(t,A,B)")

A = np.linspace(0,2,21)
B = np.linspace(-0.2,0,21)
E = np.zeros((len(A),len(B)))
Y,Z = np.meshgrid(A,B)

for i in range(len(A)):
    for j in range(len(B)):
        for k in range(len(data_arr[1])):
            E[i][j] += ((data_arr[1][k] - g(data_arr[0][k], A[i], B[j]))**2)/101

figure = plt.figure(2)
ax = figure.add_subplot(111)
Contour = ax.contour(Y,Z,E,40)
ax.clabel(Contour,[0.015,0.030,0.045,0.060,0.075] ,inline=1)
plt.title('Contour Plot',size=20)
plt.xlabel('A',size=20)
plt.ylabel('B',size=20)
plt.grid(True)
plt.legend()
plt.show()


Ea = np.empty((9,1))
Eb = np.empty((9,1))

for j in range(9):
	AB = np.linalg.lstsq(M,data_arr[j+1],rcond=None)
	Ea[j] = np.abs(AB[0][0]-p[0])
	Eb[j] = np.abs(AB[0][1]-p[1])


plt.figure(3)
plt.plot(sigma,Ea,label='Aerr',marker='o',linestyle='dashed')
plt.plot(sigma,Eb,label='Berr',marker='o',linestyle='dashed')
plt.title("Variation of Error with Noise",size=20)	
plt.xlabel('Noise standard deviation',size=20)
plt.ylabel('MS error',size=20)
plt.grid(True)
plt.legend()
plt.show()

plt.figure(4)
plt.loglog(sigma,Ea,'ro',label='Aerr')
plt.errorbar(sigma, Ea, np.std(Ea), fmt="ro")
plt.loglog(sigma,Eb,'go',label='Berr')
plt.errorbar(sigma, Eb, np.std(Eb), fmt="go")
plt.title("Variation of Error with Noise",size=20)	
plt.xlabel('Noise standard deviation',size=20)
plt.ylabel('MS error',size=20)
plt.grid(True)
plt.legend()



plt.show()