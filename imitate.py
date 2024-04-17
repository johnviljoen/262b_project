"""
This script will purely imitate the MPC solution at every timestep
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
from mpc import MPC, MPC_Vectorized

# f = dynamics.get("L_SIMO_RD1") # 24 loss
# f = dynamics.get("L_SIMO_RD2") # 124 loss
f = dynamics.get("L_SIMO_RD3") # 512 loss

@functools.partial(jax.jit, static_argnums=(2,))
def b_cost_mpc(pol_s, b_s, hzn):
    loss = 0
    Q = 10.0                # state loss
    R = 0.0001              # action loss
    b = b_s.shape[0] * hzn    # number of (s,a) pairs loss is generated over
    for _ in range(hzn):
        b_a = pol_inf(pol_s, b_s)
        b_s_kp1 = f(b_s, b_a)
        loss += (R * jnp.sum(b_a**2) + Q * jnp.sum(b_s_kp1**2)) / b
        b_s = b_s_kp1
        # if jnp.isnan(loss).any():
        #     print('fin')
    return loss

def b_cost_imitate(pol_s, b_s, hzn, mpc_vec):
    loss = 0
    b = b_s.shape[0] * hzn    # number of (s,a) pairs loss is generated over
    for _ in range(hzn):
        b_a = pol_inf(pol_s, b_s)
        b_mpc_a = mpc_vec(jax.lax.stop_gradient(b_s))
        b_s_kp1 = f(b_s, b_a)
        loss += (jnp.sum(b_a**2 - b_mpc_a)) / b
        b_s = b_s_kp1
    return loss

def cost(pol_s, s, hzn):
    loss = 0
    Q = 10.0                # state loss
    R = 0.0001              # action loss
    b = hzn                 # number of (s,a) pairs loss is generated over
    for _ in range(hzn):
        a = pol_inf(pol_s, s)
        s_kp1 = f(s, a)
        loss += (R * jnp.sum(a**2) + Q * jnp.sum(s_kp1**2)) / b
        s = s_kp1
    return loss

def step_dpc(step, opt_s, b_s, hzn=1):
    pol_s = opt_s[0]
    loss, grads = jax.value_and_grad(b_cost_mpc)(pol_s, b_s, hzn)
    grads = clip_grad_norm(grads, max_norm=100.0)
    opt_s = opt_update(step, grads, opt_s)
    return loss, opt_s

def step_imitate(step, opt_s, b_s, mpc_vec, hzn=1):
    pol_s = opt_s[0]
    loss, grads = jax.value_and_grad(b_cost_imitate)(pol_s, b_s, hzn, mpc_vec)
    grads = clip_grad_norm(grads, max_norm=100.0)
    opt_s = opt_update(step, grads, opt_s)
    return loss, opt_s

if __name__ == "__main__":

    nx, nu = 3, 1
    lr = 1e-2
    layer_sizes = [nx, 20, 20, 20, 20, nu]
    pol_s = init_pol(layer_sizes, parent_key)

    opt_init, opt_update, get_params = adagrad(lr)
    opt_s = opt_init(pol_s) # opt_s = (pol_s, optimizer_state) i.e. all variables that change, get_params gets pol_s

    batch_size = 3333
    train_data = 3.0 * np.random.randn(1, batch_size, nx)
    num_epochs_dpc = 400 # 400
    num_epochs_imitate = 10
    hzn = 4

    for epoch in range(num_epochs_dpc):
        for b_s in train_data:  # shape = (1, 3333, 2)
            loss, opt_s = step_dpc(epoch, opt_s, b_s, hzn=hzn)
        print(f"epoch: {epoch}, loss: {loss}")

    # refresh nn optimizer parameters
    pol_s = get_params(opt_s)
    opt_s = opt_init(pol_s)
    mpc_vec = MPC_Vectorized(N=hzn, nx=nx, nu=nu, ny=3, f=f, b=train_data.shape[1])

    for epoch in range(num_epochs_imitate):
        for b_s in train_data:  # shape = (1, 3333, 2)
            loss, opt_s = step_imitate(epoch, opt_s, b_s, mpc_vec, hzn=hzn)
        print(f"epoch: {epoch}, loss: {loss}")

    pol_s = get_params(opt_s)
    opt_s = opt_init(pol_s)

    for epoch in range(num_epochs_dpc):
        for b_s in train_data:  # shape = (1, 3333, 2)
            loss, opt_s = step_dpc(epoch, opt_s, b_s, hzn=hzn)
        print(f"epoch: {epoch}, loss: {loss}")

    # hzn = 10
    # for epoch in range(num_epochs):
    #     for b_s in train_data:  # shape = (1, 3333, 2)
    #         loss, opt_s = step(epoch, opt_s, b_s, hzn=hzn)
    #     print(f"epoch: {epoch}, loss: {loss}")

    # eval_data = 3.0 * np.random.randn(1, nx) generated the below:
    eval_data = jnp.array([1.59609801, 1.51405802, 4.63639117])
    eval_hzn = 10
    eval_loss = b_cost_mpc(pol_s, eval_data, eval_hzn)

    Q = 10.0                # state loss
    R = 0.0001              # action loss
    b = b_s.shape[0] * eval_hzn    # number of (s,a) pairs loss is generated over
    b_a_N, b_s_N = [], []
    for _ in range(eval_hzn):
        b_a = pol_inf(pol_s, b_s)
        b_s_kp1 = f(b_s, b_a)
        loss += (R * jnp.sum(b_a**2) + Q * jnp.sum(b_s_kp1**2)) / b
        b_s = b_s_kp1
        b_a_N.append(b_a); b_s_N.append(b_s)
    
    b_a_N, b_s_N = jnp.stack(b_a_N), jnp.stack(b_s_N)
    (R * jnp.sum(b_a_N**2) + Q * jnp.sum(b_s_N**2)) / b
    print(f'dpc loss: {loss}')

    ### MPC
    mpc = MPC(N=hzn, nx=nx, nu=nu, ny=3, f=f)
    # Q = 10.0                # state loss
    # R = 0.0001              # action loss
    u_N, x_N = [], []
    x = eval_data
    loss = 0.0
    for _ in range(10):
        u = mpc(x)
        x_kp1 = f(x, u)
        loss += (R * jnp.sum(u**2) + Q * jnp.sum(x_kp1**2)) / hzn
        x = x_kp1
        u_N.append(u); x_N.append(x)
    
    u_N, x_N = jnp.stack(u_N), jnp.stack(x_N)
    (R * jnp.sum(u_N**2) + Q * jnp.sum(x_N**2)) / hzn
    print(f'mpc loss: {loss}')

    ### MPC Vectorized
    mpc_vec = MPC_Vectorized(N=hzn, nx=nx, nu=nu, ny=3, f=f, b=train_data.shape[1])
    # Q = 10.0                # state loss
    # R = 0.0001              # action loss
    b_a_N, b_s_N = [], []
    for _ in range(eval_hzn):
        b_a = mpc_vec(b_s)
        b_s_kp1 = f(b_s, b_a)
        loss += (R * jnp.sum(b_a**2) + Q * jnp.sum(b_s_kp1**2)) / b
        b_s = b_s_kp1
        b_a_N.append(b_a); b_s_N.append(b_s)
    
    b_a_N, b_s_N = jnp.stack(b_a_N), jnp.stack(b_s_N)
    (R * jnp.sum(b_a_N**2) + Q * jnp.sum(b_s_N**2)) / b
    print(f'mpc vec loss: {loss}')

    print('fin')

    ### interesting!! we have seen that even the MPC needs a horizon of longer 
    # than 3 to achieve stability in its third state. This makes perfect sense!!
    # as the system is relative degree 3, the predictive step needs to be at least
    # as many steps as relative degrees to control all states!!!. Therefore 4 is
    # required!!

    # The problem is that we only can have a finite cost DPC with a small number of steps
    # at least initially. Also the DPC fails more and more the further along the 
    # chain of integrators we get:
    # b_s_N[-1, 1, :]
    # Array([225.7935   ,  44.549297 ,   3.4590864], dtype=float32)
    # b_s_N[-1, 10, :]
    # Array([67.69244   ,  9.557059  , -0.13979149], dtype=float32)
    # b_s_N[-1, 11, :]
    # Array([-71.5085   , -19.557402 ,  -2.7800481], dtype=float32)

    # MPC doesnt have this problem, so lets see if we can imitate to get some of its
    # performance benefits!

    


