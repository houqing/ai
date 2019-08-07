#!/usr/bin/env python3

import sys
import numpy as np

fi = sys.argv[1]
fo = fi + "--fp16"

data = np.fromfile(fi, dtype=np.float32)
data = data.astype(np.float16)
data.tofile(fo)

