import jax

seed = 0
parent_key = jax.random.PRNGKey(seed)

from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import functools
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from utils.mlp import init_pol, pol_inf
from utils.opt import adagrad, adam, clip_grad_norm
from mpc_ca import MPCVariableSpaceHorizon, generate_variable_timesteps

def gen_dataset(nb, cs):
    angles = np.random.normal(loc=1.25 * np.pi, scale=2, size=(nb,))
    x = np.sqrt(8) * np.cos(angles)
    y = np.sqrt(8) * np.sin(angles)
    xdot, ydot = np.zeros(nb), np.zeros(nb)
    s = np.stack((x, y, xdot, ydot), axis=1) + 0.01 * np.random.randn(nb, 4)
    return np.hstack([s, posVel2Cyl(s, cs)])

def gen_dataset_imitation(nb, cs, mpc):
    angles = np.random.normal(loc=1.25 * np.pi, scale=2, size=(nb,))
    x = np.sqrt(8) * np.cos(angles)
    y = np.sqrt(8) * np.sin(angles)
    xdot, ydot = np.zeros(nb), np.zeros(nb)
    s = np.stack((x, y, xdot, ydot), axis=1) + 0.01 * np.random.randn(nb, 4)
    mpc_s_hist = []
    mpc_a_hist = []
    print('generating MPC dataset - takes 3 mins or so...')
    for s_ in tqdm(s):
        _, mpc_s, mpc_a = mpc(s_, np.array([[0,0,0,0]]*(hzn+1)).T)
        mpc_s_hist.append(mpc_s[:,:-1]); mpc_a_hist.append(mpc_a[:,:-1]) # final input plays no role - is zero always in prediction
    mpc_s_hist = np.hstack(mpc_s_hist); mpc_a_hist = np.hstack(mpc_a_hist)
    cs = np.vstack([cs[0]]*mpc_s_hist.shape[1])
    return np.vstack([mpc_s_hist, posVel2Cyl(mpc_s_hist.T, cs).T]).T, mpc_a_hist.T

def posVel2Cyl(s, cs, eps=1e-10):
    # expected: s = {x, y, xd, yd}; cs = {x, y, r}; o = {xc, xcd}
    dist2center = jnp.sqrt((s[:,0:1] - cs[:,0:1])**2 + (s[:,1:2] - cs[:,1:2])**2)
    dist2cyl = dist2center - cs[:,2:3]
    vel2cyl = (s[:,0:1] - cs[:,0:1]) / (dist2center + eps) * s[:,2:3] + (s[:,1:2] - cs[:,1:2]) / (dist2center + eps) * s[:,3:4]
    return jnp.hstack([dist2cyl, vel2cyl])

def f(s, a, cs=jnp.array([[-1.0, -1.0, 0.5]]), Ts=0.1):
    # s = {x, y, xd, yd, xc, xcd}; o = {x, y, xd, yd, xc, xcd}
    A = jnp.array([
        [1.0, 0.0, Ts,  0.0],
        [0.0, 1.0, 0.0, Ts ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    B = jnp.array([
        [0.0, 0.0], 
        [0.0, 0.0],
        [Ts, 0.0],
        [0.0, Ts]
    ])
    s_next = (s[:,:4] @ A.T + a @ B.T)
    return jnp.hstack([s_next, jnp.vstack([posVel2Cyl(s_next, cs)])])

def f_pure(s, a, Ts=0.1):
    # s = {x, y, xd, yd, xc, xcd}; o = {x, y, xd, yd, xc, xcd}
    A = jnp.array([
        [1.0, 0.0, Ts,  0.0],
        [0.0, 1.0, 0.0, Ts ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    B = jnp.array([
        [0.0, 0.0], 
        [0.0, 0.0],
        [Ts, 0.0],
        [0.0, Ts]
    ])
    s_next = (s @ A.T + a @ B.T)
    return s_next

def g(s, a):
    return - s[:,4] # <= 0

log10_barrier    = lambda value: -jnp.log10(-value)
log_barrier      = lambda value: -jnp.log(-value)
inverse_barrier  = lambda value: 1 / (-value)
softexp_barrier  = lambda value, alpha=0.05: (jnp.exp(alpha * value) - 1) / alpha + alpha
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
def cost(pol_s, s, cs, hzn): # mu opt: 50k ish, standard start is 1m
    loss, Q, R, mu, b = 0, 5.0, 0.1, 1_000_000.0, s.shape[0]
    for _ in range(hzn):
        a = pol_inf(pol_s, s)
        s_next = f(s, a, cs)
        J = R * jnp.sum(a**2) + Q * jnp.sum(s_next[:,:4]**2)
        pg = barrier_cost(mu, s, a)
        loss += (J + pg) / b
        s = s_next
    return loss

@jax.jit
def cost_imitation(pol_s, s): # mu opt: 50k ish, standard start is 1m
    b = s[0].shape[0]
    return jnp.sum((s[1] - pol_inf(pol_s, s[0]))**2) / b

def step(step, opt_s, s, cs, hzn, opt_update, get_params):
    pol_s = get_params(opt_s)
    # test = cost(pol_s, s, cs, hzn)
    # test2 = jax.grad(cost)(pol_s, s, cs, hzn)
    loss, grads = jax.value_and_grad(cost)(pol_s, s, cs, hzn)
    grads = clip_grad_norm(grads, max_norm=100.0)
    # grads = jnp.nan_to_num(grads, nan=0.0)
    opt_s = opt_update(step, grads, opt_s)
    return loss, opt_s, grads

def step_imitation(step, opt_s, s, cs, hzn, opt_update, get_params, mpc):
    pol_s = get_params(opt_s)
    # test = cost(pol_s, s, cs, hzn)
    # test2 = jax.grad(cost)(pol_s, s, cs, hzn)
    loss, grads = jax.value_and_grad(cost_imitation)(pol_s, s, cs, hzn, mpc)
    grads = clip_grad_norm(grads, max_norm=100.0)
    # grads = jnp.nan_to_num(grads, nan=0.0)
    opt_s = opt_update(step, grads, opt_s)
    return loss, opt_s, grads

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from utils.plotting import plot_training_trajectory, plot_nn_contours

    ns, no, ncs, na = 4, 2, 3, 2            # state, cylinder observation, cylinder state, input sizes
    hzn, nb, nmb, ne = 20, 3333, 1, 400     # horizon, batch, minibatch, epoch sizes
    lr = 5e-3                               # learning rate
    # nb = 10 # testing
    # ne = 2
    layer_sizes = [ns + no, 20, 20, 20, 20, na]
    pol_s = init_pol(layer_sizes, parent_key)

    opt_init, opt_update, get_params = adam(lr)
    opt_s = opt_init(pol_s)

    train_cs = np.array([[-1,-1,0.5]]*nb) # np.random.randn(nb, ncs)
    train_s = gen_dataset(nb, train_cs)

    best_loss = jnp.inf
    pol_s_hist = []
    grads_hist = []
    for e in range(ne):
        loss, opt_s, grads = step(e, opt_s, train_s, train_cs, hzn, opt_update, get_params)
        leaves, treedef = jax.tree.flatten(get_params(opt_s))
        shapes_and_dtypes = [(leaf.shape, leaf.dtype) for leaf in leaves]
        pol_s_hist.append(jnp.concatenate([jnp.ravel(leaf) for leaf in leaves]))
        if loss < best_loss:
            best_opt_s = opt_s
            best_loss = loss
            print('new best:')
        print(f"epoch: {e}, loss: {loss}")
        grads, _ = jax.tree.flatten(grads)
        grads_hist.append(jnp.concatenate([jnp.ravel(leaf) for leaf in grads]))

    pol_s_hist = np.vstack(pol_s_hist)
    grads_hist = np.vstack(grads_hist)

    # plot_training_trajectory(opt_s, pol_s_hist, cost, get_params, shapes_and_dtypes, treedef, train_s, train_cs, hzn, save_name='data/combined_nn_optimization.pdf')

    # reset adam before imitation
    pol_s = get_params(best_opt_s)
    opt_s = opt_init(pol_s)

    print('performing imitation phase:')

    dts_init = generate_variable_timesteps(Ts=0.1, Tf_hzn=0.1*hzn, N=hzn)
    mpc = MPCVariableSpaceHorizon(N=hzn, Ts_0=0.1, Tf_hzn=0.1*hzn, dts_init=dts_init)
    train_imitation_s = gen_dataset_imitation(nb, train_cs, mpc)

    plot_nn_contours(
        opt_s, 
        pol_s_hist, 
        lambda pol_s, s, cs, hzn: cost_imitation(pol_s, s, cs, hzn, mpc), 
        get_params, 
        shapes_and_dtypes, 
        treedef, 
        train_imitation_s, 
        train_cs, 
        hzn, 
        save_name='data/combined_nn_optimization_imitation.pdf',
        ne=ne
    )

    best_loss = jnp.inf
    pol_s_hist_imitate = []
    for e in range(ne):
        loss, opt_s, grads = step_imitation(e, opt_s, train_imitation_s, train_cs, hzn, opt_update, get_params, mpc)
        leaves, treedef = jax.tree.flatten(get_params(opt_s))
        shapes_and_dtypes = [(leaf.shape, leaf.dtype) for leaf in leaves]
        pol_s_hist_imitate.append(jnp.concatenate([jnp.ravel(leaf) for leaf in leaves]))
        if loss < best_loss:
            best_opt_s = opt_s
            best_loss = loss
            print('new best:')
        print(f"epoch: {e}, loss: {loss}")

    pol_s_hist_imitate = np.vstack(pol_s_hist_imitate)

    print('fin')

    # n_step = 0
    # mb_len = int(nb/nmb)
    # for e in range(ne):
    #     for mb in range(mb_len):
    #         mb_s = train_s[mb_len*mb:mb_len*(mb+1)]
    #         mb_cs = train_cs[mb_len*mb:mb_len*(mb+1)]
    #         loss, opt_s = step(n_step, opt_s, mb_s, mb_cs, hzn, opt_update, get_params)
    #     if loss < best_loss:
    #         best_opt_s = opt_s
    #         best_loss = loss
    #         print('new best:')
    #     print(f"epoch: {e}, loss: {loss}")

    # plot an inferenced trajectory with start end points
    nb = 30

    cs = np.array([[-1,-1,0.5]]*nb) # np.random.randn(nb, ncs)
    s = gen_dataset(nb, cs)

    s_hist, a_hist = [], []
    pol_s = get_params(best_opt_s)
    for i, t in enumerate(range(1000)):
        a = pol_inf(pol_s, s)
        s = f(s, a, cs)
        s_hist.append(s); a_hist.append(a)
    
    s_hist = np.stack(s_hist)
    fig, ax = plt.subplots()
    for i in range(nb):
        ax.plot(s_hist[:,i,0], s_hist[:,i,1])
    ax.add_patch(Circle(cs[0,:2], cs[0,2]))
    ax.set_aspect('equal')
    plt.show()

    print('fin')