"""
 File: quick_plot.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020, AFD Group at UIUC
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

################################################################################
#                                                                              #
#  PLOT ONE PRIMITIVE                                                          #
#                                                                              #
################################################################################

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cProfile

# TODO package interface...
import pyHARM
import pyHARM.ana.plot as pplt
from pyHARM import pretty
from pyHARM.ana.units import get_units_M87
import pyHARM.parameters as parameters

# TODO parse these instead of hard-coding
USEARRSPACE = True

if not USEARRSPACE:
    SIZE = 50
    #window = (0, SIZE, 0, SIZE)
    window = (-SIZE, SIZE, -SIZE, SIZE)
    # window=(-SIZE/4, SIZE/4, 0, SIZE)
else:
    window = (0, 1, 0, 1)
    #window = (-0.1, 1.1, -0.1, 1.1)

pdf_window = (-10, 0)
FIGX = 10
FIGY = 10

dumpfile = sys.argv[1]
parfile = sys.argv[2]
var = sys.argv[3]
# Optionally take extra name, otherwise just set it to var
name = sys.argv[-1]

if len(sys.argv) > 5:
    munit = float(sys.argv[4])
    cgs = get_units_M87(munit)
    print("Uisng M_unit: ", munit)
    unit = cgs[sys.argv[3]]
    print("Will multiply by unit {} with value {}".format(sys.argv[3], unit))
    name = var + "_units"
else:
    unit = 1

#params = {'include_ghost': True}
params = {}
parameters.parse_parthenon_dat(params, parfile)
parameters.fix(params)
dump = pyHARM.load_dump(dumpfile, params=params)

# Plot vectors in 4-pane layout
# fig = plt.figure(figsize=(FIGX, FIGY))
# plt.title(pretty(var))

# if var in ['jcon', 'jcov', 'ucon', 'ucov', 'bcon', 'bcov']:
#     axes = [plt.subplot(2, 2, i) for i in range(1, 5)]
#     for n in range(4):
#         pplt.plot_xy(axes[n], dump, np.log10(dump[var][n] * unit), arrayspace=USEARRSPACE, window=window)
# elif "pdf_" in var:
#     fig = plt.figure(figsize=(FIGX, FIGY))
#     d_var, d_var_bins = dump[var]
#     plt.plot(d_var_bins[:-1], d_var)
#     if "_log_" in var:
#         plt.xlabel("Log10 value")
#     elif "_ln_" in var:
#         plt.xlabel("Ln value")
#     else:
#         plt.xlabel("Value")
#     plt.ylabel("Frequency")

#     plt.savefig(name+".png", dpi=100)
#     plt.close(fig)
#     exit() # We already saved the figure, we don't need another
# else:
#     # TODO allow specifying vmin/max, average from command line or above
#     ax = plt.subplot(1, 1, 1)
#     pplt.plot_xy(ax, dump, dump[var] * unit, log=False, arrayspace=USEARRSPACE, window=window)

# plt.tight_layout()
# plt.savefig(name + "_xy.png", dpi=100)
# plt.close(fig)

# Plot XZ
fig = plt.figure(figsize=(FIGX, FIGY))

if var in ['jcon', 'jcov', 'ucon', 'ucov', 'bcon', 'bcov']:
    axes = [plt.subplot(2, 2, i) for i in range(1, 5)]
    for n in range(4):
        pplt.plot_xz(axes[n], dump, np.log10(dump[var][n] * unit), arrayspace=USEARRSPACE, window=window)
else:
    ax = plt.subplot(1, 1, 1)
    pplt.plot_xz(ax, dump, dump[var] * unit, log=False, arrayspace=USEARRSPACE, window=window)
    #pplt.overlay_field(ax, dump, nlines=5, arrayspace=USEARRSPACE)

plt.tight_layout()

plt.savefig(name + "_xz.png", dpi=100)
plt.close(fig)
