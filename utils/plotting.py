import jax.numpy as jnp
import jax
import numpy as np
from utils.geometry import project_vector, find_coefficients, pca
from utils.jax import reconstruct_pytree    
import matplotlib.pyplot as plt

def plot_training_trajectory(opt_s, pol_s_hist, cost, get_params, shapes_and_dtypes, treedef, train_s, train_cs, hzn):
    print('performing PCA analysis plotting')
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
    
    losses = np.clip(losses, a_min=0.0, a_max = 2_000)

    # Create a figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surface = ax.plot_surface(X, Y, np.log(losses), cmap='viridis')
    # line = ax.plot(xy[:,0], xy[:,1], np.log(losses_traj), color='red')

    # Add a color bar which maps values to colors
    fig.colorbar(surface, shrink=0.5, aspect=5)

    # Set labels and title
    ax.set_xlabel('PC 2 axis')
    ax.set_ylabel('PC 1 axis')
    ax.set_zlabel('Loss axis')

    # Show the plot
    plt.savefig('data/nn_optimization_landscape.pdf')
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
    plt.xlabel('PC 2')
    plt.ylabel('PC 1')
    plt.legend()

    plt.savefig('data/nn_optimization_trajectory.pdf')