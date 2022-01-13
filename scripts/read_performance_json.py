#!/usr/bin/env python3

import sys
import json
import re
import numpy as np

top_kernels = 10

for fname in sys.argv[1:]:
    f = open(fname, "r")

    contents = ''.join(f.readlines())
    # Add the things Kokkos missed
    contents = "{" + contents.replace('kokkos-kernel-data :', '"kokkos-kernel-data" :') + "}"
    # Take out the extra commas
    contents = re.sub('\,(?=\s*?[\}\]])', '', contents)
    # THEN parse
    kperf = json.loads(contents)['kokkos-kernel-data']

    print("=== FILE: {} ===".format(fname))
    print("Total time: {} kernels: {} non-kernels: {} ({}% in kernels)".format(
            kperf['total-app-time'], kperf['total-kernel-times'], kperf['total-non-kernel-times'], kperf['percent-in-kernels']))

    print("Top {} kernels:".format(top_kernels))
    top_kernel_info = sorted(kperf['kernel-perf-info'], key=lambda x: x['total-time'], reverse=True)[:10]
    for x in top_kernel_info:
        print(" * {}: {}s total".format(x['kernel-name'], x['total-time']))
