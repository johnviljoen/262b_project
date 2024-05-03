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
from utils.opt import adagrad, adam, clip_grad_norm

def posVel2Cyl(s, cs, eps=1e-10):
    # expected: s = {x, y, xd, yd}; cs = {x, y, r}; o = {xc, xcd}
    dist2center = jnp.sqrt((s[:,0:1] - cs[:,0:1])**2 + (s[:,1:2] - cs[:,1:2])**2)
    dist2cyl = dist2center - cs[:,2:3]
    vel2cyl = (s[:,0:1] - cs[:,0:1]) / (dist2center + eps) * s[:,2:3] + (s[:,1:2] - cs[:,1:2]) / (dist2center + eps) * s[:,3:4]
    return jnp.hstack([dist2cyl, vel2cyl])

def f(s, a, cs=jnp.array([[1.0, 1.0, 0.5]])):
    # s = {x, y, xd, yd, xc, xcd}; o = {x, y, xd, yd, xc, xcd}
    A = jnp.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    B = jnp.array([
        [0.0, 0.0], 
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    s_next = (s[:,:4] @ A.T + a @ B.T)
    return jnp.hstack([s_next, jnp.vstack([posVel2Cyl(s_next, cs)])])

def g(s, a):
    return - s[:,4] # <= 0

log10_barrier    = lambda value: -jnp.log10(-value)
log_barrier      = lambda value: -jnp.log(-value)
inverse_barrier  = lambda value: 1 / (-value)
softexp_barrier  = lambda value, alpha=0.5: (jnp.exp(alpha * value) - 1) / alpha + alpha
softlog_barrier  = lambda value, alpha=0.05: -jnp.log(1 + alpha * (-value - alpha)) / alpha
expshift_barrier = lambda value, shift=1.0: jnp.exp(value + shift)

def barrier_cost(multiplier, s, a, barrier=log_barrier, upper_bound=1.0):
    cvalue = g(s, a)
    penalty_mask = cvalue > 0
    cviolation = jnp.clip(cvalue, a_min=0.0) # could just use penalty mask too
    cbarrier = barrier(cvalue)
    cbarrier = jnp.nan_to_num(cbarrier, nan=0.0, posinf=0.0)
    cbarrier = jnp.clip(cbarrier, a_min=0.0, a_max=upper_bound)
    penalty_loss = multiplier * jnp.mean(penalty_mask * cviolation)
    barrier_loss = multiplier * jnp.mean(~penalty_mask * cbarrier)
    return penalty_loss + barrier_loss

@functools.partial(jax.jit, static_argnums=(3,))
def cost(pol_s, s, cs, hzn):
    loss, Q, R, mu, b = 0, 10.0, 0.0001, 1_000_000.0, s.shape[0]
    for _ in range(hzn):
        a = pol_inf(pol_s, s)
        s_next = f(s, a, cs)
        J = R * jnp.sum(a**2) + Q * jnp.sum(s_next[:,:4]**2)
        pg = barrier_cost(mu, s, a)
        loss += (J + pg) / b
        s = s_next
    return loss

def step(step, opt_s, s, cs, hzn, opt_update, get_params):
    pol_s = get_params(opt_s)
    loss, grads = jax.value_and_grad(cost)(pol_s, s, cs, hzn)
    grads = clip_grad_norm(grads, max_norm=100.0)
    opt_s = opt_update(step, grads, opt_s)
    return loss, opt_s

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    ns, no, ncs, na = 4, 2, 3, 2            # state, cylinder observation, cylinder state, input sizes
    hzn, nb, nmb, ne = 10, 3333, 1, 600     # horizon, batch, minibatch, epoch sizes
    lr = 1e-3                               # learning rate

    layer_sizes = [ns + no, 20, 20, 20, 20, na]
    pol_s = init_pol(layer_sizes, parent_key)

    opt_init, opt_update, get_params = adam(lr)
    opt_s = opt_init(pol_s)

    train_s = 3 * np.random.randn(nb, ns)
    train_cs = np.array([[1,1,0.5]]*nb)# np.random.randn(nb, ncs)
    train_s = np.hstack([train_s, posVel2Cyl(train_s, train_cs)])
    valid_mask = train_s[:,4] >= 0
    train_s, train_cs = train_s[valid_mask], train_cs[valid_mask]

    best_loss = jnp.inf
    for e in range(ne):
        loss, opt_s = step(e, opt_s, train_s, train_cs, hzn, opt_update, get_params)
        if loss < best_loss:
            best_opt_s = opt_s
            best_loss = loss
            print('new best:')
        print(f"epoch: {e}, loss: {loss}")

    # plot an inferenced trajectory with start end points
    nb = 20

    s = 3 * np.random.randn(nb, ns)
    cs = np.array([[1,1,0.5]]*nb)# np.random.randn(nb, ncs)
    s = np.hstack([s, posVel2Cyl(s, cs)])
    mask = s[:,4] >= 0
    s, cs = s[mask], cs[mask]
    s_hist, a_hist = [], []
    pol_s = get_params(best_opt_s)
    for i, t in enumerate(range(100)):
        a = pol_inf(pol_s, s)
        s = f(s, a, cs)
        s_hist.append(s); a_hist.append(a)
    
    s_hist = np.vstack(s_hist)
    fig, ax = plt.subplots()
    ax.plot(s_hist[:,0], s_hist[:,1])
    ax.add_patch(Circle(cs[0,:2],cs[0,2]))
    ax.set_aspect('equal')
    plt.show()

    print('fin')