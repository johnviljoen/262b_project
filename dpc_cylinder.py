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

    ns, no, ncs, na = 4, 2, 3, 2            # state, cylinder observation, cylinder state, input sizes
    hzn, nb, nmb, ne = 20, 3333, 1, 400     # horizon, batch, minibatch, epoch sizes
    lr = 5e-3                               # learning rate

    layer_sizes = [ns + no, 20, 20, 20, 20, na]
    pol_s = init_pol(layer_sizes, parent_key)

    opt_init, opt_update, get_params = adam(lr)
    opt_s = opt_init(pol_s)

    train_cs = np.array([[-1,-1,0.5]]*nb) # np.random.randn(nb, ncs)
    train_s = gen_dataset(nb, train_cs)

    # Function to reconstruct the original leaves from the concatenated vector
    def reconstruct_pytree(vector, shapes_and_dtypes, treedef):
        offset = 0
        new_leaves = []
        for shape, dtype in shapes_and_dtypes:
            size = np.prod(shape)
            new_leaves.append(vector[offset:offset+size].reshape(shape).astype(dtype))
            offset += size
        new_pytree = jax.tree.unflatten(treedef, new_leaves)
        # assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: jnp.allclose(x, y), original_pytree, new_pytree))
        return new_pytree
    
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

    # https://medium.com/@nahmed3536/a-python-implementation-of-pca-with-numpy-1bbd3b21de2e
    def pca(data, num_components=2):
        standardized_data = (data - data.mean(axis = 0)) / data.std(axis = 0)
        covariance_matrix = np.cov(standardized_data, ddof = 1, rowvar = False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
        order_of_importance = np.argsort(eigenvalues)[::-1] 
        # utilize the sort order to sort eigenvalues and eigenvectors
        # sorted_eigenvalues = eigenvalues[order_of_importance]
        sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns
        # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
        # explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        reduced_data = standardized_data @ sorted_eigenvectors[:,:num_components] # transform the original data
        # total_explained_variance = sum(explained_variance[:k])
        return reduced_data
    
    components = pca(pol_s_hist.T)
    vec_best_pol_s, _ = jax.tree.flatten(get_params(opt_s))
    vec_best_pol_s = jnp.concatenate([jnp.ravel(leaf) for leaf in vec_best_pol_s])
    test_unknown_pytree = reconstruct_pytree(vec_best_pol_s + 0.001 * components[:,0] + 0.001 * components[:,1], shapes_and_dtypes, treedef)
    test_pytree = reconstruct_pytree(vec_best_pol_s, shapes_and_dtypes, treedef)
    test_s, test_cs, test_hzn = train_s[0:1,:], train_cs[0:1,:], hzn
    loss = cost(test_pytree, test_s, test_cs, test_hzn)

    def project_vector(vec_to_project, base_vec, components):
        """
        Project a vector onto the hyperplane defined by a base vector and components.

        Parameters:
        - vec_to_project: The vector to be projected.
        - base_vec: A point on the hyperplane (e.g., vec_best_pol_s).
        - components: A matrix whose columns define directions spanning the hyperplane.

        Returns:
        - projected_vec: The projection of vec_to_project onto the hyperplane.
        """
        B = components
        projector = B @ jnp.linalg.inv(B.T @ B) @ B.T
        projection = projector @ (vec_to_project - base_vec)
        projected_vec = base_vec + projection
        return projected_vec
    
    project_vector(vec_best_pol_s + 0.001 * components[:,0] + 0.001 * components[:,1], vec_best_pol_s, components)

    # Generate 100 points from -0.01 to 0.01
    x = np.linspace(-0.01, 0.01, 100)
    y = np.linspace(-0.01, 0.01, 100)

    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    losses = np.zeros_like(X)
    for i in range(100):
        for j in range(100):
            vec = vec_best_pol_s + X[i,j] * components[:,0] + Y[i,j] * components[:,1]
            pytree = reconstruct_pytree(vec, shapes_and_dtypes, treedef)
            losses[i,j] = cost(pytree, train_s, train_cs, hzn)

    # Create a figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surface = ax.plot_surface(X, Y, losses, cmap='viridis')

    # Add a color bar which maps values to colors
    fig.colorbar(surface, shrink=0.5, aspect=5)

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Surface Plot')

    # Show the plot
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