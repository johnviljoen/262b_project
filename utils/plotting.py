import jax.numpy as jnp
import jax
import numpy as np
from utils.geometry import project_vector, find_coefficients, pca
from utils.jax import reconstruct_pytree    
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LogNorm
from matplotlib import animation
from datetime import datetime
import os
from matplotlib.patches import Circle

def plot_training_trajectory(opt_s, pol_s_hist, cost, get_params, shapes_and_dtypes, treedef, train_s, train_cs, train_im_s, hzn, save_name='data/combined_nn_optimization.pdf', ne=400):
    print('performing PCA analysis plotting...')
    components = pca(pol_s_hist.T)
    vec_best_pol_s, _ = jax.tree.flatten(get_params(opt_s))
    vec_best_pol_s = jnp.concatenate([jnp.ravel(leaf) for leaf in vec_best_pol_s])

    projected_vecs = np.zeros_like(pol_s_hist)
    coeffs = np.zeros([ne,3])
    xy = np.zeros([ne,2])
    losses_traj = np.zeros([ne])
    losses_true_traj = np.zeros([ne])
    for i in range(ne):
        projected_vecs[i] = project_vector(pol_s_hist[i], vec_best_pol_s, components)
        coeffs[i] = find_coefficients(vec_best_pol_s, components, projected_vecs[i])
        xy[i] = coeffs[i,1:]
        projected_vecs_pytree = reconstruct_pytree(projected_vecs[i], shapes_and_dtypes, treedef)
        true_vecs_pytree = reconstruct_pytree(pol_s_hist[i], shapes_and_dtypes, treedef)
        losses_traj[i] = cost(projected_vecs_pytree, train_s, train_cs, train_im_s, hzn)
        losses_true_traj[i] = cost(true_vecs_pytree, train_s, train_cs, train_im_s,hzn)

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
    X, Y = np.meshgrid(x, y)

    losses = np.zeros_like(X)
    print('gathering losses...')
    for i in tqdm(range(num_points)):
        for j in range(num_points):
            vec = vec_best_pol_s + X[i,j] * components[:,0] + Y[i,j] * components[:,1]
            pytree = reconstruct_pytree(vec, shapes_and_dtypes, treedef)
            losses[i,j] = cost(pytree, train_s, train_cs, hzn)
    
    losses_clipped = np.clip(losses, a_min=0.0, a_max = 2_000)

    # Create a figure with two subplots
    fig = plt.figure(figsize=(25, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Plot the 3D surface on the first subplot
    surface = ax1.plot_surface(X, Y, np.log(losses_clipped), cmap='viridis')
    # fig.colorbar(surface, ax=ax1, shrink=0.5, aspect=10)
    ax1.set_xlabel('PC 2')
    ax1.set_ylabel('PC 1')
    ax1.set_zlabel('Log Loss')

    # Plot the 2D contour plot on the second subplot
    contour = ax2.contourf(X, Y, np.log(losses_clipped), levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax2)
    contour_lines = ax2.contour(X, Y, np.log(losses_clipped), colors='black', linestyles='dashed', linewidths=1)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f')
    ax2.plot(xy[:,0], xy[:,1], 'r-', label='Optimization Trajectory')
    ax2.legend()
    ax2.set_xlabel('PC 2')
    ax2.set_ylabel('PC 1')

    # Plot the 2D contour plot on the second subplot
    contour_noclip = ax3.contourf(X, Y, np.log(losses), levels=50, cmap='viridis')
    plt.colorbar(contour_noclip, ax=ax3)
    contour_lines_noclip = ax3.contour(X, Y, np.log(losses), colors='black', linestyles='dashed', linewidths=1)
    ax3.clabel(contour_lines_noclip, inline=True, fontsize=8, fmt='%1.1f')
    ax3.plot(xy[:,0], xy[:,1], 'r-', label='Optimization Trajectory')
    ax3.legend()
    ax3.set_xlabel('PC 2')
    ax3.set_ylabel('PC 1')

    # Save the combined plot to a file
    plt.savefig(save_name)

    plt.show()

def plot_nn_contours(opt_s, pol_s_hist, cost, get_params, shapes_and_dtypes, treedef, train_s, train_cs, hzn, save_name='data/combined_nn_optimization.pdf', ne=400):
    print('performing PCA analysis plotting...')
    components = pca(pol_s_hist.T)
    vec_best_pol_s, _ = jax.tree.flatten(get_params(opt_s))
    vec_best_pol_s = jnp.concatenate([jnp.ravel(leaf) for leaf in vec_best_pol_s])

    projected_vecs = np.zeros_like(pol_s_hist)
    coeffs = np.zeros([ne,3])
    xy = np.zeros([ne,2])
    losses_traj = np.zeros([ne])
    losses_true_traj = np.zeros([ne])
    for i in range(ne):
        projected_vecs[i] = project_vector(pol_s_hist[i], vec_best_pol_s, components)
        coeffs[i] = find_coefficients(vec_best_pol_s, components, projected_vecs[i])
        xy[i] = coeffs[i,1:]
        projected_vecs_pytree = reconstruct_pytree(projected_vecs[i], shapes_and_dtypes, treedef)
        true_vecs_pytree = reconstruct_pytree(pol_s_hist[i], shapes_and_dtypes, treedef)
        losses_traj[i] = cost(projected_vecs_pytree, train_s, train_cs, hzn)
        losses_true_traj[i] = cost(true_vecs_pytree, train_s, train_cs, hzn)

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
    X, Y = np.meshgrid(x, y)

    losses = np.zeros_like(X)
    print('gathering losses...')
    for i in tqdm(range(num_points)):
        for j in range(num_points):
            vec = vec_best_pol_s + X[i,j] * components[:,0] + Y[i,j] * components[:,1]
            pytree = reconstruct_pytree(vec, shapes_and_dtypes, treedef)
            losses[i,j] = cost(pytree, train_s, train_cs, hzn)
    
    losses_clipped = np.clip(losses, a_min=0.0, a_max = 2_000)

    # Create a figure with two subplots
    fig = plt.figure(figsize=(6, 5))
    ax2 = fig.add_subplot(111)

    # Plot the 2D contour plot on the second subplot
    contour = ax2.contourf(X, Y, np.log(losses_clipped), levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax2)
    contour_lines = ax2.contour(X, Y, np.log(losses_clipped), colors='black', linestyles='dashed', linewidths=1)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f')
    ax2.scatter(xy[-1,0], xy[-1,1], color='r', label='Current NN Parameter Vector')
    ax2.legend()
    ax2.set_xlabel('PC 2')
    ax2.set_ylabel('PC 1')

    # Save the combined plot to a file
    plt.savefig(save_name)


class Animator:
    def __init__(
            self,
            states, times, references, state_prediction = None,
            max_frames = 500,
            save_path = 'data/media', save_name = None,
    ) -> None:
        
        self.save_path = save_path
        self.save_name = save_name
        num_steps = len(times)
        max_frames = 500
        self.num_frames = 1
        self.ifsave = True
        self.dt = 0.1
        self.preds = None

        def compute_render_interval(num_steps, max_frames):
            render_interval = 1  # Start with rendering every frame.
            # While the number of frames using the current render interval exceeds max_frames, double the render interval.
            while num_steps / render_interval > max_frames:
                render_interval *= 2
            return render_interval
        
        render_interval = compute_render_interval(num_steps, max_frames)

        if state_prediction is not None:
            self.state_prediction = state_prediction[::render_interval,:]
        else:
            self.state_prediction = state_prediction

        self.states = states[::render_interval,:]
        self.times = times[::render_interval]
        self.references = references[::render_interval,:]

        # Unpack States for readability
        # -----------------------------

        self.x = self.states[:,0]
        self.y = self.states[:,1]
        self.xd = self.states[:,2]
        self.xd = self.states[:,3]

        # Instantiate the figure with title, time, limits...
        # --------------------------------------------------

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()

        self.xDes = self.references[:, 0]
        self.yDes = self.references[:, 1]
        extraEachSide = 0.5

        x_min = min(np.min(self.x), np.min(self.xDes))
        y_min = min(np.min(self.y), np.min(self.yDes))
        x_max = max(np.max(self.x), np.max(self.xDes))
        y_max = max(np.max(self.y), np.max(self.yDes))

        maxRange = 0.5*np.array([x_max-x_min, y_max-y_min]).max() + extraEachSide
        mid_x = 0.5*(x_max+x_min)
        mid_y = 0.5*(y_max+y_min)
        
        self.ax.set_xlim([mid_x-maxRange, mid_x+maxRange])
        self.ax.set_xlabel('X')
        self.ax.set_ylim([mid_y-maxRange, mid_y+maxRange])
        self.ax.set_ylabel('Y')

        self.line1, = self.ax.plot([], [], lw=2, color='red')
        self.ax.add_patch(Circle([1, 1], 0.5))

        self.ax.scatter(self.xDes, self.yDes, color='green', alpha=1, marker = 'o', s = 25)

    def draw_predictions(self, i, state_prediction):

        # predicted_x is in the form of (prediction idx, state, timestep idx)
        return self.ax.plot(state_prediction[i,0,:], state_prediction[i,1,:], color='black')

    def update_lines(self, i):

        # we draw this every self.num_frames
        time = self.times[i * self.num_frames]

        # to draw the history of the line so far we need to retrieve all of it
        x_from0 = self.x[0:i * self.num_frames]
        y_from0 = self.y[0:i * self.num_frames]

        # if using predictive control, plot the predictions
        if self.state_prediction is not None:

            if self.preds is not None:
                self.preds[0].remove()

            self.preds = self.draw_predictions(i, self.state_prediction)

        self.line1.set_data(x_from0, y_from0)

        return self.line1

    def ini_plot(self):

        self.line1.set_data(np.empty([1]), np.empty([1]))

        return self.line1
    
    def animate(self):
        line_ani = animation.FuncAnimation(
            self.fig, 
            self.update_lines, 
            init_func=self.ini_plot, 
            # frames=len(self.times[0:-2:self.num_frames]), 
            frames=len(self.times)-1, 
            interval=(self.dt*10), 
            blit=False)

        if (self.ifsave):
            if self.save_name is None:
                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                print(f"save path: {os.path.abspath(self.save_path)}")
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                line_ani.save(f'{self.save_path}/{current_datetime}.gif', dpi=120, fps=25)
                # Update the figure with the last frame of animation
                self.update_lines(len(self.times[1:])-1)
                # Save the final frame as an SVG for good paper plots
                self.fig.savefig(f'{self.save_path}/{current_datetime}_final_frame.svg', format='svg')
            else:
                line_ani.save(f'{self.save_path}/{self.save_name}.gif', dpi=120, fps=25)

        # plt.close(self.fig)            
        # plt.show()
        return line_ani
