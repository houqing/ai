#!/usr/bin/env python3

import sys
import numpy as np

fi = sys.argv[1]
fo = fi + "--fp32"

data = np.fromfile(fi, dtype=np.float16)
data = data.astype(np.float32)
data.tofile(fo)

