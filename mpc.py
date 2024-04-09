from jax import jit, grad, jacrev, jacfwd
import jax.numpy as jnp
import dynamics
from cyipopt import minimize_ipopt

class MPC:
    def __init__(
        self,
        N, nx, nu, ny,
        f
        ) -> None:

        self.N, self.nx, self.nu, self.ny = N, nx, nu, ny
        self.f = f
        self.Q = jnp.ones(nx)
        self.R = jnp.ones(nu)

        # optimization variable memory
        self.z_init = jnp.zeros(N*nx + (N-1)*nu)

        # jit the functions
        self.l_jit = jit(self.l)
        self.h_jit = jit(self.h)
        self.g_jit = jit(self.g)

        # build the derivatives and jit them
        self.l_grad = jit(grad(self.l_jit))  # objective gradient
        self.l_hess = jit(jacrev(jacfwd(self.l_jit))) # objective hessian
        self.h_jac = jit(jacfwd(self.h_jit))  # jacobian
        self.g_jac = jit(jacfwd(self.g_jit))  # jacobian
        h_hess = jacrev(jacfwd(self.h_jit)) # hessian
        self.h_hessvp = jit(lambda x, v: jnp.sum(h_hess(x) * v[:, None, None], axis=0))
        g_hess = jacrev(jacfwd(self.g_jit))  # hessian
        self.g_hessvp = jit(lambda x, v: jnp.sum(g_hess(x) * v[:, None, None], axis=0))

        # Constraint definitions for IPOPT
        self.constraints = [
            {'type': 'eq', 'fun': self.h_jit, 'jac': self.h_jac, 'hess': self.h_hessvp},
            {'type': 'ineq', 'fun': self.g_jit, 'jac': self.g_jac, 'hess': self.g_hessvp}
        ]

        # call minimize once for cython compilation
        result = minimize_ipopt(
            fun=self.l_jit, 
            x0=self.z_init, 
            jac=self.l_grad, 
            hess=self.l_hess, 
            constraints=self.constraints, 
            tol=1e-6, options={'disp': 5, 'maxiter': 100}
        )

    def __call__(self, x):
        pass

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
    mpc = MPC(
        N=3, nx=3, nu=1, ny=3, f=f
    )

    print('fin')