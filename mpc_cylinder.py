from dpc_cylinder import gen_dataset, posVel2Cyl, f, g
from mpc import MPC_Vectorized_Cylinder
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

hzn = 10
nx = 4
nu = 2
nb = 30
mpc = MPC_Vectorized_Cylinder(hzn, nx, nu, 6, f, nb)

cs = np.array([[-1,-1,0.5]]*nb) # np.random.randn(nb, ncs)
s = gen_dataset(nb, cs)

s_hist, a_hist = [], []
for i, t in enumerate(range(1000)):
    a = mpc(s)
    s = f(s, a, cs)
    s_hist.append(s); a_hist.append(a)

s_hist = np.stack(s_hist)
fig, ax = plt.subplots()
for i in range(nb):
    ax.plot(s_hist[:,i,0], s_hist[:,i,1])
ax.add_patch(Circle(cs[0,:2], cs[0,2]))
ax.set_aspect('equal')
plt.show()