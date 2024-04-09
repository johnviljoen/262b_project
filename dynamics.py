"""
This file will represent some interesting dynamics that scale from trivial linear
SIMO systems to complex nonlinear MIMO systems. DPC will begin to fail in some,
and MPC with IPOPT will succeed in others, and some will be unsolveable for IPOPT.
"""

import jax.numpy as jnp

def get(name="L_SIMO_RD1"):

    def L_SIMO_RD1(x, u):
        A = jnp.array([
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        B = jnp.array([
            [1.0], 
            [0.5]
        ])
        return (x @ A.T + u @ B.T) # @ C.T
    
    def L_SIMO_RD2(x, u):
        A = jnp.array([
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        B = jnp.array([
            [0.0], 
            [0.5]
        ])
        return (x @ A.T + u @ B.T) # @ C.T
    
    def L_SIMO_RD3(x, u):
        A = jnp.array([
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]
        ])
        B = jnp.array([
            [0.0], 
            [0.0],
            [1.0]
        ])
        return x @ A.T + u @ B.T
    
    def L_SIMO_RD4(x, u):
        A = jnp.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        B = jnp.array([
            [0.0], 
            [0.0],
            [0.0],
            [1.0]
        ])
        return (x @ A.T + u @ B.T)        

    def L_MISO():
        pass

    def L_MIMO():
        pass

    def NL_SISO():
        pass

    def NL_SIMO():
        pass

    def NL_MISO():
        pass

    def NL_MIMO():
        pass

    def PVRD_NL_SISO():
        pass

    def NL_SIMO_PVRD(x, u):
        x_1_kp1 = x[0] + x[0]**2 * x[1]
        x_2_kp1 = x[1] + u
        x_next = jnp.hstack([x_1_kp1, x_2_kp1])
        return x_next

    def PVRD_NL_MISO():
        pass

    def PVRD_NL_MIMO():
        pass

    def p3():
        print("tip3")

    # Access the local function dictionary
    funcs = locals()
    # Select and return the function by name, if exists
    if name in funcs:
        return funcs[name]
    else:
        raise ValueError(f"No function named {name} found.")
        
