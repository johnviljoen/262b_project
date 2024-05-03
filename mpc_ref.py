from jax import jit, grad, jacrev, jacfwd
import jax.numpy as jnp
import dynamics
from cyipopt import minimize_ipopt
import numpy as np
from tqdm import tqdm

class MPC_Vectorized:

    """
    Forms the MPC problem and solves it of form:

        min sum { x.T Q x + u.T @ R u }
        x,u
        s.t. h(x, u) = 0
             g(x, u) < 0
    
    """

    def __init__(
            self,
            N, nx, nu, ny,
            f, b=1,
        ) -> None:

        self.N, self.nx, self.nu, self.ny = N, nx, nu, ny
        self.f = f
        self.b = b
        self.Q = jnp.ones(nx)
        self.R = jnp.ones(nu)

        # optimization variable memory
        self.z_init = jnp.zeros([b, N*nx + (N-1)*nu])
        x0 = jnp.zeros(nx) # parameter for initial condition

        # jit the functions
        self.l_jit = jit(self.l)
        self.h_jit = jit(self.h)
        self.g_jit = jit(self.g)

        # build the derivatives and jit them
        self.l_grad = jit(grad(self.l_jit))  # objective gradient
        self.l_hess = jit(jacrev(jacfwd(self.l_jit))) # objective hessian
        self.h_jac = jit(jacfwd(self.h_jit))  # jacobian
        self.h_hess = jacrev(jacfwd(self.h_jit)) # hessian
        # self.h_hessvp = jit(lambda x, v: jnp.sum(h_hess(x) * v[:, None, None], axis=0))
        self.g_jac = jit(jacfwd(self.g_jit))  # jacobian
        g_hess = jacrev(jacfwd(self.g_jit))  # hessian
        self.g_hessvp = jit(lambda x, v: jnp.sum(g_hess(x) * v[:, None, None], axis=0))

        # equality constraint definitions need to be updated with changed values for current x
        h_jit = lambda z: self.h_jit(z, x0) 
        h_jac = lambda z: self.h_jac(z, x0)
        h_hessvp = lambda z, v: jnp.sum(self.h_hess(z, x0) * v[:, None, None], axis=0)

        constraints = [
            {'type': 'eq', 'fun': h_jit, 'jac': h_jac, 'hess': h_hessvp},
            {'type': 'ineq', 'fun': self.g_jit, 'jac': self.g_jac, 'hess': self.g_hessvp}
        ]

        # call minimize once for cython compilation
        result = minimize_ipopt(
            fun=self.l_jit,
            x0=self.z_init[0],
            jac=self.l_grad,
            hess=self.l_hess,
            constraints=constraints,
            tol=1e-6,
            options={'disp': 0, 'maxiter': 100}
        )

        print('fin')

    def __call__(self, x_b):

        # equality constraint definitions need to be updated with changed values for current x
        # this implicit x application is disgusting but necessary to work with the jax jitted
        # functions and cyipopt which cannot pass arguments to the constraints beyond  the 
        # optimization variables, there is a clean way to do this by creating another function
        # which passes *args and returns a new constraints list, but I think overcomplicated 
        # for just an MPC like this.
        z_init = []
        sol_u = []
        for i, x in tqdm(enumerate(x_b)): # every initial condition in the batch input

            h_jit = lambda z: self.h_jit(z, x) 
            h_jac = lambda z: self.h_jac(z, x)
            h_hessvp = lambda z, v: jnp.sum(self.h_hess(z, x) * v[:, None, None], axis=0)

            constraints = [
                {'type': 'eq', 'fun': h_jit, 'jac': h_jac, 'hess': h_hessvp},
                {'type': 'ineq', 'fun': self.g_jit, 'jac': self.g_jac, 'hess': self.g_hessvp}
            ]
            
            # call minimize once for cython compilation
            result = minimize_ipopt(
                fun=self.l_jit,
                x0=self.z_init[i], # warm start
                jac=self.l_grad,
                hess=self.l_hess,
                constraints=constraints, 
                tol=1e-6, options={'disp': 0, 'maxiter': 100}
            )

            # Extract solution and assign warm start z_init
            z_init.append(result.x)
            # warm = result.x[self.nx:self.N*self.nx]

            sol_x = result.x[:self.N*self.nx].reshape(self.N, self.nx)
            sol_u.append(result.x[self.N*self.nx:].reshape(self.N-1, self.nu))
            # print('Optimal state trajectory:', sol_x)
            # print('Optimal control inputs:', sol_u)

        self.z_init = jnp.array(z_init)
        sol_u = jnp.array(sol_u)

        return sol_u[:,0,:]

    # loss: l
    def l(self, z):
        # unpack
        x = z[:self.N*self.nx].reshape(self.N, self.nx)
        u = z[self.N*self.nx:].reshape(self.N-1, self.nu)
        # return quadratic cost
        return jnp.sum(self.Q * x**2) + jnp.sum(self.R * (u**2))

    # equality constraints: h
    def h(self, z, x0):
        x = z[:self.N*self.nx].reshape(self.N, self.nx)
        u = z[self.N*self.nx:].reshape(self.N-1, self.nu)

        cl = []
        # Enforce initial condition
        cl.append(x[0] - x0)

        # Enforce system dynamics as equality constraints
        for k in range(self.N-1):
            x_next = self.f(x[k], u[k])
            cl.append(x[k+1] - x_next)

        return jnp.concatenate(cl).flatten()
    
    # inequality constraints: g
    def g(self, z):
        x = z[:self.N*self.nx].reshape(self.N, self.nx)
        u = z[self.N*self.nx:].reshape(self.N-1, self.nu)

        cl = [] # constraint list

        # random inequalities - can be nonlinear ofc
        cl.append(x[0] + 25)       # = 0
        cl.append(x[1] + 25)   # = 0
        cl.append(u[0] + 25)       # = 0
        cl.append(u[1] + 25)   # = 0

        # stacking of equality constraints
        return jnp.hstack(cl) 


class MPC:

    """
    Forms the MPC problem and solves it of form:

        min sum { x.T Q x + u.T @ R u }
        x,u
        s.t. h(x, u) = 0
             g(x, u) < 0
    
    """

    def __init__(
            self,
            N, nx, nu, ny,
            f,
        ) -> None:

        self.N, self.nx, self.nu, self.ny = N, nx, nu, ny
        self.f = f
        self.Q = jnp.ones(nx)
        self.R = jnp.ones(nu)

        # optimization variable memory
        self.z_init = jnp.zeros(N*nx + (N-1)*nu)
        x0 = jnp.zeros(nx) # parameter for initial condition

        # jit the functions
        self.l_jit = jit(self.l)
        self.h_jit = jit(self.h)
        self.g_jit = jit(self.g)

        # build the derivatives and jit them
        self.l_grad = jit(grad(self.l_jit))  # objective gradient
        self.l_hess = jit(jacrev(jacfwd(self.l_jit))) # objective hessian
        self.h_jac = jit(jacfwd(self.h_jit))  # jacobian
        self.h_hess = jacrev(jacfwd(self.h_jit)) # hessian
        # self.h_hessvp = jit(lambda x, v: jnp.sum(h_hess(x) * v[:, None, None], axis=0))
        self.g_jac = jit(jacfwd(self.g_jit))  # jacobian
        g_hess = jacrev(jacfwd(self.g_jit))  # hessian
        self.g_hessvp = jit(lambda x, v: jnp.sum(g_hess(x) * v[:, None, None], axis=0))

        # equality constraint definitions need to be updated with changed values for current x
        h_jit = lambda z: self.h_jit(z, x0) 
        h_jac = lambda z: self.h_jac(z, x0)
        h_hessvp = lambda z, v: jnp.sum(self.h_hess(z, x0) * v[:, None, None], axis=0)

        constraints = [
            {'type': 'eq', 'fun': h_jit, 'jac': h_jac, 'hess': h_hessvp},
            {'type': 'ineq', 'fun': self.g_jit, 'jac': self.g_jac, 'hess': self.g_hessvp}
        ]

        # call minimize once for cython compilation
        result = minimize_ipopt(
            fun=self.l_jit,
            x0=self.z_init,
            jac=self.l_grad,
            hess=self.l_hess,
            constraints=constraints,
            tol=1e-6,
            options={'disp': 0, 'maxiter': 100}
        )

        print('fin')

    def __call__(self, x):

        # equality constraint definitions need to be updated with changed values for current x
        # this implicit x application is disgusting but necessary to work with the jax jitted
        # functions and cyipopt which cannot pass arguments to the constraints beyond  the 
        # optimization variables, there is a clean way to do this by creating another function
        # which passes *args and returns a new constraints list, but I think overcomplicated 
        # for just an MPC like this.

        h_jit = lambda z: self.h_jit(z, x) 
        h_jac = lambda z: self.h_jac(z, x)
        h_hessvp = lambda z, v: jnp.sum(self.h_hess(z, x) * v[:, None, None], axis=0)

        constraints = [
            {'type': 'eq', 'fun': h_jit, 'jac': h_jac, 'hess': h_hessvp},
            {'type': 'ineq', 'fun': self.g_jit, 'jac': self.g_jac, 'hess': self.g_hessvp}
        ]
        
        # call minimize once for cython compilation
        result = minimize_ipopt(
            fun=self.l_jit,
            x0=self.z_init,
            jac=self.l_grad,
            hess=self.l_hess,
            constraints=constraints, 
            tol=1e-6, options={'disp': 0, 'maxiter': 100}
        )

        # Extract solution and assign warm start z_init
        self.z_init = result.x
        # warm = result.x[self.nx:self.N*self.nx]

        sol_x = result.x[:self.N*self.nx].reshape(self.N, self.nx)
        sol_u = result.x[self.N*self.nx:].reshape(self.N-1, self.nu)
        # print('Optimal state trajectory:', sol_x)
        # print('Optimal control inputs:', sol_u)

        return sol_u[0]

    # loss: l
    def l(self, z):
        # unpack
        x = z[:self.N*self.nx].reshape(self.N, self.nx)
        u = z[self.N*self.nx:].reshape(self.N-1, self.nu)
        # return quadratic cost
        return jnp.sum(self.Q * x**2) + jnp.sum(self.R * (u**2))

    # equality constraints: h
    def h(self, z, x0):
        x = z[:self.N*self.nx].reshape(self.N, self.nx)
        u = z[self.N*self.nx:].reshape(self.N-1, self.nu)

        cl = []
        # Enforce initial condition
        cl.append(x[0] - x0)

        # Enforce system dynamics as equality constraints
        for k in range(self.N-1):
            x_next = self.f(x[k], u[k])
            cl.append(x[k+1] - x_next)

        return jnp.concatenate(cl).flatten()
    
    # inequality constraints: g
    def g(self, z):
        x = z[:self.N*self.nx].reshape(self.N, self.nx)
        u = z[self.N*self.nx:].reshape(self.N-1, self.nu)

        cl = [] # constraint list

        # random inequalities - can be nonlinear ofc
        cl.append(x[0] + 25)       # = 0
        cl.append(x[1] + 25)   # = 0
        cl.append(u[0] + 25)       # = 0
        cl.append(u[1] + 25)   # = 0

        # stacking of equality constraints
        return jnp.hstack(cl) 
    

    
if __name__ == "__main__":
    
    f = dynamics.get("L_SIMO_RD3")
    N = 10
    nx = 3
    nu = 1
    mpc = MPC(N=N, nx=nx, nu=nu, ny=3, f=f)

    x0 = jnp.zeros(nx) # parameter for initial condition
    x1 = jnp.ones(nx)
    mpc(x0)
    mpc(x1)

    x_vec = jnp.stack([x0, x1])
    mpc_vec = MPC_Vectorized(N=N, nx=nx, nu=nu, ny=3, f=f, b=len(x_vec))
    mpc_vec(x_vec)

    # eval_data = 3.0 * np.random.randn(nx) generated the below
    eval_data = jnp.array([1.59609801, 1.51405802, 4.63639117])

    Q = 10.0                # state loss
    R = 0.0001              # action loss
    u_N, x_N = [], []
    x = eval_data
    loss = 0.0
    for _ in range(10):
        u = mpc(x)
        x_kp1 = f(x, u)
        loss += (R * jnp.sum(u**2) + Q * jnp.sum(x_kp1**2)) / N
        x = x_kp1
        u_N.append(u); x_N.append(x)
    
    u_N, x_N = jnp.stack(u_N), jnp.stack(x_N)
    (R * jnp.sum(u_N**2) + Q * jnp.sum(x_N**2)) / N
    print(f'loss: {loss}')


    print('fin')