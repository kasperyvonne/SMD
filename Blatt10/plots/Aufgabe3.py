yimport matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_hdf('image_parameters_smd_reduced.hdf5')
print(data.keys())
#.values macht nparray mit inhalten
y = data.corsika_run_header_particle_id

y1 = y[y == 1]
y0 = y[y == 14]

print(len(y))
print(y)
