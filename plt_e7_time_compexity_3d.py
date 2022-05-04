"""
Experiment 7 - measure time complexity of method
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp2d

chunk_sizes = [50,100,150,200,250]
features=[10,20,30,40,50]
subspace_sizes = [1,2]

res = np.load('results_ex7/time.npy')
#replications, chunk_sizes, features, subspace_sizes
res = res[:,:,:,0]

mean_res = np.mean(res, axis=0)

fig = plt.figure()
ax = fig.gca(projection='3d')

x = features
y = chunk_sizes
x, y = np.meshgrid(x, y)
z = mean_res

f = interp2d(x, y, z, kind='cubic')
q=20
new_x = np.linspace(features[0], features[-1], q)
new_y = np.linspace(chunk_sizes[0], chunk_sizes[-1], q)
new_z = f(new_x, new_y)

new_x, new_y = np.meshgrid(new_x, new_y)

surf = ax.plot_wireframe(new_x, new_y, new_z,
                       linewidth=0.1, antialiased=False)

ax.view_init(45,235)

# plt.show()
plt.savefig('foo.png')