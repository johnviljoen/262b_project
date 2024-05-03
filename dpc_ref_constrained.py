"""
This script demonstrates DPCs failure on a system, which IPOPT succeeds on in ipopt.py

whats more is that without the horizon being tiny at 2 the loss will become infinite and 
training will stall.
"""

import jax

seed = 0
parent_key = jax.random.PRNGKey(seed)

from jax import config
config.update("jax_enable_x64", False)
config.update('jax_platform_name', 'cpu')

import functools
import numpy as np
import jax.numpy as jnp

from utils.mlp import init_pol, pol_inf
from utils.opt import adagrad, clip_grad_norm
import dynamics

# barriers taken directly from neuromancer
shift = 1.
alpha = 0.05
barriers = {
    'log10': lambda value: -jnp.log10(-value),
    'log': lambda value: -jnp.log(-value),
    'inverse': lambda value: 1 / (-value),
    'softexp': lambda value: (jnp.exp(alpha * value) - 1) / alpha + alpha,
    'softlog': lambda value: -jnp.log(1 + alpha * (-value - alpha)) / alpha,
    'expshift': lambda value: jnp.exp(value + shift)
}
barrier = barriers['softexp']
Q_con = 1_000_000

f = dynamics.get("L_MIMO_RD2") # double integrator in 2D (4 states 2 inputs)

def h(s, a): # equality constraints enforced in f
    pass

def g(s, a):
    # enforce cylinder constraints
    x, xd = posVel2Cyl(s)
    pass

def posVel2Cyl(s, cs, eps=1e-10):
    # expected: s = {x, y, xd, yd}; cs = {x, y, r}
    dist2center = jnp.sqrt((s[:,0] - cs[:,0])**2 + (s[:,1] - cs[:,1])**2)
    dist2cyl = dist2center - cs[:,2]
    vel2cyl = (s[:,0] - cs[:,0]) / (dist2center + eps) * s[:,2] + (s[:,1] - cs[:,1]) / (dist2center + eps) * s[:,3]
    return dist2cyl, vel2cyl

def get_data(nx, nc, b):
    state_data = 3.0 * np.random.randn(b, nx)
    cyl_pos, cyl_vel = posVel2Cyl(state_data, 3.0 * np.random.randn(b, 3))

@functools.partial(jax.jit, static_argnums=(2,3))
def b_cost(pol_s, b_s_cs, hzn, nx):
    loss = 0
    Q = 10.0                # state loss
    R = 0.0001              # action loss
    b = b_s_cs.shape[0] * hzn    # number of (s,a) pairs loss is generated over
    mu = 1.0
    for _ in range(hzn):
        b_a = pol_inf(pol_s, b_s_cs)
        b_s = b_s_cs[:,:nx]
        b_s_kp1 = f(b_s, b_a)
        J = R * jnp.sum(b_a**2) + Q * jnp.sum(b_s_kp1**2)
        b_s_cs_kp1 = jnp.hstack([b_s_kp1, b_s_cs[:,nx:]])
        pg = mu @ jnp.log(g(b_s_cs_kp1, b_a)) # penalty on inequality constraints g
        ph = jnp.zeros_like(pg) # eq constraint h automatically enforced here
        loss += (J + ph + pg) / b # dual loss
        b_s_cs = b_s_cs_kp1
        # if jnp.isnan(loss).any():
        #     print('fin')
    return loss

def cost(pol_s, s, hzn):
    loss = 0
    Q = 10.0                # state loss
    R = 0.0001              # action loss
    b = hzn                 # number of (s,a) pairs loss is generated over
    mu = 1.0
    for _ in range(hzn):
        a = pol_inf(pol_s, s)
        s_kp1 = f(s, a)
        J = R * jnp.sum(a**2) + Q * jnp.sum(s_kp1**2)
        pg = mu @ - jnp.log(-g(s_kp1, a)) # penalty on inequality constraints g
        ph = jnp.zeros_like(pg) # eq constraint h automatically enforced here
        loss += (J + ph + pg) / b
        s = s_kp1
    return loss

def b_step(step, opt_s, b_s, hzn=1, nx=4):
    pol_s = opt_s[0]
    loss, grads = jax.value_and_grad(b_cost)(pol_s, b_s, hzn, nx)
    grads = clip_grad_norm(grads, max_norm=100.0)
    opt_s = opt_update(step, grads, opt_s)
    return loss, opt_s

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    nx, nu, nc = 4, 2, 2
    lr = 1e-2
    layer_sizes = [nx + nc, 20, 20, 20, 20, nu]
    pol_s = init_pol(layer_sizes, parent_key)

    opt_init, opt_update, get_params = adagrad(lr)
    opt_s = opt_init(pol_s) # opt_s = (pol_s, optimizer_state) i.e. all variables that change, get_params gets pol_s

    train_data = 3.0 * np.random.randn(1, 3333, nx + nc)
    num_epochs = 400 # 400
    hzn = 3

    for epoch in range(num_epochs):
        for b_s in train_data:  # shape = (1, 3333, 2)
            loss, opt_s = b_step(epoch, opt_s, b_s, hzn=hzn, nx=nx)
        print(f"epoch: {epoch}, loss: {loss}")

    hzn = 10
    for epoch in range(num_epochs):
        for b_s in train_data:  # shape = (1, 3333, 2)
            loss, opt_s = b_step(epoch, opt_s, b_s, hzn=hzn, nx=nx)
        print(f"epoch: {epoch}, loss: {loss}")

    eval_data = 3.0 * np.random.randn(3333, nx)
    eval_hzn = 10
    eval_loss = b_cost(pol_s, eval_data, eval_hzn)

    # plot an inferenced trajectory with start end points
    s = np.array([1,2,3,4.])
    s_hist, a_hist = [], []
    pol_s = get_params(opt_s)
    for i, t in enumerate(range(100)):
        a = pol_inf(pol_s, s)
        s = f(s, a)
        s_hist.append(s); a_hist.append(a)
    
    s_hist = np.vstack(s_hist)
    plt.plot(s_hist[:,0], s_hist[:,1])


    print('fin')

