import jax

seed = 0
parent_key = jax.random.PRNGKey(seed)

from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import functools
import numpy as np
import jax.numpy as jnp

from utils.mlp import init_pol, pol_inf
from utils.opt import adagrad, adam, clip_grad_norm

def gen_dataset(nb, cs):
    angles = np.random.normal(loc=1.25 * np.pi, scale=2, size=(nb,))
    x = np.sqrt(8) * np.cos(angles)
    y = np.sqrt(8) * np.sin(angles)
    xdot, ydot = np.zeros(nb), np.zeros(nb)
    s = np.stack((x, y, xdot, ydot), axis=1) + 0.01 * np.random.randn(nb, 4)
    return np.hstack([s, posVel2Cyl(s, cs)])

def posVel2Cyl(s, cs, eps=1e-10):
    # expected: s = {x, y, xd, yd}; cs = {x, y, r}; o = {xc, xcd}
    dist2center = jnp.sqrt((s[:,0:1] - cs[:,0:1])**2 + (s[:,1:2] - cs[:,1:2])**2)
    dist2cyl = dist2center - cs[:,2:3]
    vel2cyl = (s[:,0:1] - cs[:,0:1]) / (dist2center + eps) * s[:,2:3] + (s[:,1:2] - cs[:,1:2]) / (dist2center + eps) * s[:,3:4]
    return jnp.hstack([dist2cyl, vel2cyl])

def f(s, a, cs=jnp.array([[1.0, 1.0, 0.5]]), Ts=0.1):
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

def step(step, opt_s, s, cs, hzn, opt_update, get_params):
    pol_s = get_params(opt_s)
    # test = cost(pol_s, s, cs, hzn)
    # test2 = jax.grad(cost)(pol_s, s, cs, hzn)
    loss, grads = jax.value_and_grad(cost)(pol_s, s, cs, hzn)
    grads = clip_grad_norm(grads, max_norm=100.0)
    # grads = jnp.nan_to_num(grads, nan=0.0)
    opt_s = opt_update(step, grads, opt_s)
    return loss, opt_s

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from utils.jax import reconstruct_pytree
    from utils.geometry import pca, project_vector, find_coefficients

    ns, no, ncs, na = 4, 2, 3, 2            # state, cylinder observation, cylinder state, input sizes
    hzn, nb, nmb, ne = 20, 3333, 1, 400     # horizon, batch, minibatch, epoch sizes
    lr = 5e-3                               # learning rate

    layer_sizes = [ns + no, 20, 20, 20, 20, na]
    pol_s = init_pol(layer_sizes, parent_key)

    opt_init, opt_update, get_params = adam(lr)
    opt_s = opt_init(pol_s)

    train_cs = np.array([[-1,-1,0.5]]*nb) # np.random.randn(nb, ncs)
    train_s = gen_dataset(nb, train_cs)

    best_loss = jnp.inf
    pol_s_hist = []
    for e in range(ne):
        loss, opt_s = step(e, opt_s, train_s, train_cs, hzn, opt_update, get_params)
        leaves, treedef = jax.tree.flatten(get_params(opt_s))
        shapes_and_dtypes = [(leaf.shape, leaf.dtype) for leaf in leaves]
        pol_s_hist.append(jnp.concatenate([jnp.ravel(leaf) for leaf in leaves]))
        if loss < best_loss:
            best_opt_s = opt_s
            best_loss = loss
            print('new best:')
        print(f"epoch: {e}, loss: {loss}")

    pol_s_hist = np.vstack(pol_s_hist)
    
    components = pca(pol_s_hist.T)
    vec_best_pol_s, _ = jax.tree.flatten(get_params(opt_s))
    vec_best_pol_s = jnp.concatenate([jnp.ravel(leaf) for leaf in vec_best_pol_s])

    projected_vecs = np.zeros_like(pol_s_hist)
    coeffs = np.zeros([400,3])
    xy = np.zeros([400,2])
    losses_traj = np.zeros([400])
    losses_true_traj = np.zeros([400])
    for i in range(400):
        projected_vecs[i] = project_vector(pol_s_hist[i], vec_best_pol_s, components)
        coeffs[i] = find_coefficients(vec_best_pol_s, components, projected_vecs[i])
        xy[i] = coeffs[i,1:]
        projected_vecs_pytree = reconstruct_pytree(projected_vecs[i], shapes_and_dtypes, treedef)
        true_vecs_pytree = reconstruct_pytree(pol_s_hist[i], shapes_and_dtypes, treedef)
        losses_traj[i] = cost(projected_vecs_pytree, train_s, train_cs, hzn)
        losses_true_traj[i] = cost(true_vecs_pytree, train_s, train_cs, hzn)

    # Generate 100 points on each axis
    num_points = 100

    x_delta = xy[:,0].max() - xy[:,0].min()
    y_delta = xy[:,1].max() - xy[:,1].min()
    x_max = max(np.abs(xy[:,0].min() - x_delta*0.1), np.abs(xy[:,0].max() + x_delta*0.1))
    x_min = -x_max
    
    y_min = xy[:,1].min() - y_delta*0.1
    y_max = xy[:,1].max() + y_delta*0.1    
    y_max = max(np.abs(xy[:,1].min() - y_delta*0.1), np.abs(xy[:,1].max() + y_delta*0.1))
    y_min = -y_max

    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)

    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    losses = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            vec = vec_best_pol_s + X[i,j] * components[:,0] + Y[i,j] * components[:,1]
            pytree = reconstruct_pytree(vec, shapes_and_dtypes, treedef)
            losses[i,j] = cost(pytree, train_s, train_cs, hzn)
    
    # losses = np.clip(losses, a_min=0.0, a_max = 2_000)

    # # Create a figure and a 3D subplot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the surface
    # surface = ax.plot_surface(X, Y, np.log(losses), cmap='viridis')
    # line = ax.plot(xy[:,0], xy[:,1], np.log(losses_traj), color='red')

    # # Add a color bar which maps values to colors
    # fig.colorbar(surface, shrink=0.5, aspect=5)

    # # Set labels and title
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # ax.set_title('3D Surface Plot')

    # # Show the plot
    # plt.show()

    # Create the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, np.log(losses), cmap='viridis', levels=50)
    plt.colorbar(contour)
    contour_lines = plt.contour(X, Y, np.log(losses), colors='black', linestyles='dashed', linewidths=1)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f')

    # Plot the trajectory
    plt.plot(xy[:,0], xy[:,1], 'r-', label='Optimization Trajectory')
    # plt.scatter(xy[:,0], xy[:,1], c='red')  # Optionally mark points
    plt.xlabel('PCA 2')
    plt.ylabel('PCA 1')
    plt.legend()

    plt.savefig('nn_optimization_trajectory_noclip.pdf')
    plt.show()


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