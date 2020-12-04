#!/usr/bin/env python3

import matplotlib.pyplot as plt

fil = open("convergence.txt")
lines = fil.readlines()

xn = [16, 32, 64]
prim_names = ["rho", "u", "u1", "u2", "u3", "B1", "B2", "B3"]

plt.figure()
nmax = 0
for col in range(len(lines[1].split()) - 2):
    n = [float(lines[row].split()[2+col].strip("[]")) for row in range(len(lines))]
    if n[0] > nmax:
        nmax = n[0]
    plt.loglog(xn, n, label=prim_names[col])
plt.loglog(xn, [nmax * (x**-2/xn[0]**-2) for x in xn], 'k--', label="x^-2")
plt.legend(loc='upper right')
plt.savefig("convergence.png")