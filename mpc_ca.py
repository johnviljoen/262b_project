import casadi as ca
import numpy as np

def generate_variable_timesteps(Ts, Tf_hzn, N):

    # Find the optimal dts for the MPC
    dt_1 = Ts
    d = (2 * (Tf_hzn/N) - 2 * dt_1) / (N - 1)
    dts = [dt_1 + i * d for i in range(N)]
    return dts

class MPC:
    def __init__(
            self,
            N, Ts, f, xT
        ) -> None:

        n, m = 4, 2
        xc, yc, rc = -1, -1, 0.5

        self.opti = ca.Opti()
        self.X = self.opti.variable(n, N+1) # first state plays no role
        self.U = self.opti.variable(m, N+1) # final input plays no role
        self.x0 = self.opti.parameter(n, 1)

        Q = np.diag([1,1,1,1.]) * 5
        R = np.diag([1,1.]) * 0.1

        # box
        for k in range(N+1):
            self.opti.subject_to(self.X[:,k] < np.array([3,3,3,3]))
            self.opti.subject_to(self.X[:,k] > np.array([-3,-3,-3,-3]))
            self.opti.subject_to(self.U[:,k] < np.array([1,1]))
            self.opti.subject_to(self.U[:,k] > np.array([-1,-1]))

        # dynamics
        for k in range(N):
            self.opti.subject_to(self.X[:,k+1] == self.X[:,k] + f(self.X[:,k], self.U[:,k]) * Ts)
        
        # initial conditions
        self.opti.subject_to(self.X[:,0] == self.x0)

        # terminal conditions
        # self.opti.subject_to(self.X[:,-1] == xT)

        # cylinder constraint
        for k in range(N-1):
            # apply the constraint from 2 timesteps in the future as the quad has relative degree 2
            # to ensure it is always feasible!
            current_time = ca.sum1(k * Ts)
            multiplier = 1 + current_time * 0.1
            self.opti.subject_to(rc ** 2 * multiplier <= (self.X[0,k+2] - xc)**2 + (self.X[1,k+2] - yc)**2)

        self.opti.solver('ipopt', {'ipopt.print_level':0, 'print_time':0, 'ipopt.tol': 1e-6})
        cost = ca.MX(0)
        for k in range(N+1):
            cost += self.X[:,k].T @ Q @ self.X[:,k] + self.U[:,k].T @ R @ self.U[:,k]
        self.opti.minimize(cost)

        # test solve
        self.x_sol, self.u_sol = np.zeros([n,N+1]), np.zeros([m,N+1])
        self.opti.set_value(self.x0, np.array([0, 0, 0, 0]))
        self.opti.set_initial(self.X, self.x_sol)
        self.opti.set_initial(self.U, self.u_sol)
        sol = self.opti.solve()
        self.x_sol, self.u_sol = sol.value(self.X), sol.value(self.U)

    def __call__(self, x):

        self.opti.set_value(self.x0, x)
        self.opti.set_initial(self.X, self.x_sol)
        self.opti.set_initial(self.U, self.u_sol)

        sol = self.opti.solve()
        self.x_sol, self.u_sol = sol.value(self.X), sol.value(self.U)
        return self.u_sol[:,0], self.x_sol
    

class MPCVariableSpaceHorizon:
    def __init__(
            self,
            N,                          # Prediction Horizon No. Inputs
            Ts_0,                       # Timestep of Simulation
            Tf_hzn,                     # Final time prediction horizon reaches
            dts_init,                   # initial variable timestep
            integrator_type = "euler",  # "euler", "RK4"
        ):

        self.N = N
        self.Ts = Ts_0
        self.Tf_hzn = Tf_hzn
        self.dts_init = dts_init
        self.state_ub = np.array([30,30,30,30])
        self.state_lb = np.array([-30,-30,-30,-30])
        self.integrator_type = integrator_type

        self.n, self.m = 4, 2

        # create optimizer and define its optimization variables
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.n, N+1)
        self.U = self.opti.variable(self.m, N+1) # final input plays no role

        # adaptive timestep means that it must be seen as a parameter to the opti
        self.dts = self.opti.parameter(self.N)
        self.opti.set_value(self.dts, self.dts_init)

        # create Q, R
        self.Q, self.R = self.create_weight_matrices()

        # apply the dynamics constraints over timesteps defined in dts
        self.apply_dynamics_constraints(self.opti, self.dts)

        # apply the state and input constraints
        self.apply_state_input_constraints(self.opti)

        # apply the cylinder constraint if demanded across variable timestep
        self.apply_cylinder_constraint(self.opti, self.dts)

        # solver setup
        opts = {
            'ipopt.print_level':0, 
            'print_time':0,
            'ipopt.tol': 1e-6,
        } # silence!
        self.opti.solver('ipopt', opts)

        # define start condition (dummy)
        state0 = np.array([0,0,0,0])
        self.init = self.opti.parameter(self.n,1)
        self.opti.set_value(self.init, state0)
        self.opti.subject_to(self.X[:,0] == self.init)

        # define dummy reference (n x N)
        reference = np.array([[2,2,0,0]]*(N+1)).T
        self.ref = self.opti.parameter(self.n,N+1)
        self.opti.set_value(self.ref, reference)

        # cost function
        self.opti.minimize(self.cost(self.X, self.ref, self.U)) # discounted

        # solve the mpc once, so that we can do it repeatedly in a method later
        sol = self.opti.solve()

        # use the initial solution as the first warm start
        self.x_sol, self.u_sol = sol.value(self.X), sol.value(self.U)
        

    def __call__(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        # point to point control, from state to stationary reference

        # define start condition
        self.opti.set_value(self.init, state)

        # warm starting
        # reference_stack = np.array([np.linspace(state[i], reference[i,-1], self.N) for i in range(self.n)])
        old_x_sol = self.x_sol[:,2:] # ignore old start and first step (this step start)
        x_warm_start = np.hstack([old_x_sol, old_x_sol[:,-1:]]) # stack final solution onto the end again for next warm start
        old_u_sol = self.u_sol[:,1:] # ignore previous solution
        u_warm_start = np.hstack([old_u_sol, old_u_sol[:,-1:]]) # stack final u solution onto the end again for next warm start

        self.opti.set_initial(self.X[:,1:], x_warm_start)
        self.opti.set_initial(self.U[:,:], u_warm_start) 

        # define cost w.r.t reference
        self.opti.set_value(self.ref, reference)

        # solve
        sol = self.opti.solve()

        # save the solved x's and u'x to warm start the next time around
        self.x_sol, self.u_sol = sol.value(self.X), sol.value(self.U)

        # return first input to be used
        return self.u_sol[:,0], self.x_sol, self.u_sol

    def cost(self, state, reference, input):
        state_error = reference - state
        cost = ca.MX(0)
        # lets get cost per timestep:
        for k in range(self.N + 1):
            timestep_input = input[:,k]
            timestep_state_error = state_error[:,k]
            cost += (timestep_state_error.T @ self.Q @ timestep_state_error + timestep_input.T @ self.R @ timestep_input)
        return cost

    # Constraints and Weights Methods 
    # -------------------------------

    def dynamics(self, x, u):
        A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
        B = np.array([
            [0.0, 0.0], 
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        return A @ x + B @ u

    def create_weight_matrices(self):

        # define weighting matrices
        Q = ca.MX.zeros(self.n,self.n)
        Q[0,0] =   1 # x
        Q[1,1] =   1 # y
        Q[2,2] =   1 # xd
        Q[3,3] =   1 # yd

        R = ca.MX.eye(self.m)

        return Q, R

    def apply_dynamics_constraints(self, opti, dts):
        # constrain optimisation to the system dynamics over the horizon
        if self.integrator_type == 'euler':
            for k in range(self.N):
                input = self.U[:,k]
                sdot_k = self.dynamics(self.X[:,k], input)
                opti.subject_to(self.X[:,k+1] == self.X[:,k] + sdot_k * dts[k])

        elif self.integrator_type == 'RK4':
            for k in range(self.N):
                k1 = self.dynamics(self.X[:,k], self.U[:,k])
                k2 = self.dynamics(self.X[:,k] + dts[k] / 2 * k1, self.U[:,k])
                k3 = self.dynamics(self.X[:,k] + dts[k] / 2 * k2, self.U[:,k])
                k4 = self.dynamics(self.X[:,k] + dts[k] * k3, self.U[:,k])
                x_next = self.X[:,k] + dts[k] / 6 * (k1 + 2*k2 + 2*k3 + k4)
                opti.subject_to(self.X[:,k+1] == x_next)

    def apply_state_input_constraints(self, opti):
        # apply state constraints
        for k in range(self.N):
            opti.subject_to(self.X[:,k] < self.state_ub)
            opti.subject_to(self.X[:,k] > self.state_lb)

        # define input constraints
        opti.subject_to(opti.bounded(-100, self.U, 100))

    def apply_cylinder_constraint(self, opti, dts):

        self.x_pos = 1
        self.y_pos = 1
        self.radius = 0.5
        
        # apply the constraint from 2 timesteps in the future as the quad has relative degree 2
        # to ensure it is always feasible!
        for k in range(self.N-1):
            current_time = ca.sum1(dts[:k])
            multiplier = 1 + current_time * 0.1
            current_x, current_y = self.X[0,k+2], self.X[1,k+2]
            opti.subject_to(self.is_in_cylinder(current_x, current_y, multiplier))

    # Utility Methods
    # ---------------

    # cylinder enlargens in the future to stop us colliding with cylinder,
    # mpc expects it to be worse than it is.
    def is_in_cylinder(self, X, Y, multiplier):
        return self.radius ** 2 * multiplier <= (X - self.x_pos)**2 + (Y - self.y_pos)**2
    

if __name__ == "__main__":

    def f(x, u):
        A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
        B = np.array([
            [0.0, 0.0], 
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        return A @ x + B @ u

    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    Ti, Ts, Tf = 0.0, 0.1, 3.5
    N, Tf_hzn = 35, 3.5

    dts_init = generate_variable_timesteps(Ts, Tf_hzn, N)

    mpc = MPCVariableSpaceHorizon(N, Ts, Tf_hzn, dts_init)

    x = np.array([0, 0, 0, 0.])
    r = np.array([[2,2,0,0]]*(N+1)).T
    x_hist = [x]
    x_preds = []
    times = np.arange(Ti, Tf, Ts)
    for t in tqdm(times):
        u, preds, u_preds = mpc(x, r)
        # print(u_preds)
        x_preds.append(preds)
        x += f(x, u) * Ts
        print(x)
        x_hist.append(np.copy(x))

    x_hist = np.vstack(x_hist)
    x_preds = np.stack(x_preds)
    fig, ax = plt.subplots()
    ax.plot(x_hist[:,0], x_hist[:,1])
    ax.add_patch(Circle([1, 1], 0.5))
    plt.show()

    from utils.plotting import Animator

    animator = Animator(x_hist[1:], times[1:], r[1:], x_preds[1:])
    animator.animate()
    
    print('fin')