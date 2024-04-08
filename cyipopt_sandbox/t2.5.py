"""
This will be a mirror of t2.py in casadi
"""

import casadi as ca
import numpy as np

def f(x, u):
    A = ca.DM([[1.0, 1.0],[0.0,1.0]])
    B = ca.DM([[0.0],[1.0]])
    x_kp1 = A @ x + B @ u
    return x_kp1

mpc_params = {
    "nx": 2,
    "nu": 1,
    "N": 2,
    "x0": np.array([1., 1.])
}

class MPC:
    def __init__(
            self,
            mpc_params
        ) -> None:
        
        self.nx, self.nu, self.N = mpc_params["nx"], mpc_params["nu"], mpc_params["N"]

        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, self.N)
        self.U = self.opti.variable(self.nu, self.N)
        self.Q = np.eye(2)
        self.R = np.eye(1)

        # apply initial_condition_constraint
        self.x0 = self.opti.parameter(self.nx, 1)

        # apply dynamics constraints
        for k in range(self.N):
            if k == 0:  # hard initial condition constraint
                self.opti.subject_to(self.X[:,k] == f(self.x0, self.U[:,k]))
            else:       # soft dynamics constraints
                self.opti.subject_to(self.X[:,k] == f(self.X[:,k-1], self.U[:,k]))

        # objective
        self.opti.minimize(self.cost(self.X, self.U))

        # solver setup
        opts = {
            'ipopt.print_level':5, 
            'print_time':1,
            'ipopt.tol': 1e-6,
        } # silence!
        self.opti.solver('ipopt', opts)


    def __call__(self, x):

        self.opti.set_value(self.x0, x)
        sol = self.opti.solve()
        return sol.value(self.X), sol.value(self.U)
    
    def cost(self, x, u):
        cost = ca.MX(0)
        for k in range(self.N):
            cost += x[:,k].T @ self.Q @ x[:,k] + u[:,k].T @ self.R @ u[:,k]
        return cost
    
mpc = MPC(mpc_params)
x0 = mpc_params["x0"]
sol_x, sol_u = mpc(x0)
print(sol_x)
print(sol_u)
print('f')