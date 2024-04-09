"""
This file will represent some interesting dynamics that scale from trivial linear
SIMO systems to complex nonlinear MIMO systems. DPC will begin to fail in some,
and MPC with IPOPT will succeed in others, and some will be unsolveable for IPOPT.
"""

import jax.numpy as jnp

def get(dynamics="L_SIMO_RD1"):

    def L_SIMO_RD1(x, u):
        A = jnp.array([
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        B = jnp.array([
            [1.0], 
            [0.5]
        ])
        # C = jnp.array([[1.0, 1.0]])
        return (x @ A.T + u @ B.T) # @ C.T
    
    if dynamics == "L_SIMO_RD1":
        return L_SIMO_RD1

    def L_SIMO_RD2(x, u):
        A = jnp.array([
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        B = jnp.array([
            [0.0], 
            [0.5]
        ])
        # C = jnp.array([[1.0, 1.0]])
        return (x @ A.T + u @ B.T) # @ C.T
    
    if dynamics == "L_SIMO_RD2":
        return L_SIMO_RD2
    
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
    
    if dynamics == "L_SIMO_RD3":
        return L_SIMO_RD3
    
    def L_SISO_RD4(x, u):
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
        C = jnp.array([[1.0, 1.0, 1.0, 1.0]])
        return C @ (A @ x + B @ u)        

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

    def PVRD_NL_SIMO():
        pass

    def PVRD_NL_MISO():
        pass

    def PVRD_NL_MIMO():
        pass

    def p3():
        print("tip3")
