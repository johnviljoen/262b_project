from dpc_cylinder import gen_dataset, posVel2Cyl, f_pure, g
from mpc import MPC_Vectorized_Cylinder
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm

hzn = 10
nx = 4
nu = 2
nb = 1
mpc = MPC_Vectorized_Cylinder(hzn, nx, nu, 4, f_pure, nb)

cs = np.array([[-1,-1,0.5]]*nb) # np.random.randn(nb, ncs)
# s = gen_dataset(nb, cs)[:,:4]
s = np.array([[-2, -2, 0, 0.]])

s_hist, a_hist = [], []
for i in tqdm(range(100)):
    a = mpc(s)
    s = f_pure(s, a)
    s_hist.append(s); a_hist.append(a)

s_hist = np.stack(s_hist)
fig, ax = plt.subplots()
for i in range(nb):
    ax.plot(s_hist[:,i,0], s_hist[:,i,1])
ax.add_patch(Circle(cs[0,:2], cs[0,2]))
ax.set_aspect('equal')
plt.show()