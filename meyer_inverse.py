import numpy as np
from nose.tools import ok_
from framework import *



# calculate the updated pseudoinverse using meyer's 6 theorems
# NOTE assuming real throughout
def meyer_update(A, Ainv, c, d):
    # pseudoinverse of vectors
    def x_dagger(x):
        return x.T / (np.linalg.norm(x)**2)

    M, N = A.shape
    # d star is just d transpose
    dt = d.T
    k = Ainv @ c
    h = dt @ Ainv
    # daggered k and h as required by the paper
    kd = x_dagger(k)
    hd = x_dagger(h)
    u = c - A @ k
    ok_(np.allclose(u, (np.eye(M) - A @ Ainv) @ c))
    v = dt - h @ A
    ok_(np.allclose(v, dt @ (np.eye(N) - Ainv @ A)))
    ud = x_dagger(u)
    vd = x_dagger(v)
    beta = 1 + dt @ Ainv @ c

    c_in_A = np.allclose(u, 0)
    d_in_At = np.allclose(v, 0)
    ok_(c_in_A == in_col_space(A, c), d_in_At == in_col_space(A.T, d))

    def inverse1():
        print("inv1")
        return Ainv - (k @ ud) - (vd @ h) + (beta * vd @ ud)

    def inverse2():
        print("inv2")
        return Ainv - (k @ kd @ Ainv) - (vd @ h)

    def inverse3():
        print("inv3")
        p1 = - (np.linalg.norm(k)**2 / beta)*v.T - k
        q1t = - (np.linalg.norm(v)**2 / beta)*k.T @ Ainv - h
        sigma1 = np.linalg.norm(k)**2 * np.linalg.norm(v)**2 + beta**2
        return Ainv + (1/beta)*v.T @ k.T @ Ainv - (beta/sigma1)* p1 @ q1t

    def inverse4():
        print("inv4")
        return Ainv - Ainv @ hd @ h - k @ ud

    def inverse5():
        print("inv5")
        p2 = - (np.linalg.norm(u)**2/beta)*Ainv @ h.T - k
        q2t = - (np.linalg.norm(h)**2/beta)*u.T - h
        sigma2 = np.linalg.norm(h)**2 * np.linalg.norm(u)**2 + beta**2
        return Ainv + (1/beta) * Ainv @ h.T @ u.T - (beta/sigma2)*p2 @ q2t

    def inverse6():
        print("inv6")
        return Ainv - (k @ kd @ Ainv) - (Ainv @ hd @ h) + ((kd @ Ainv @ hd) * k @ h)

    # c in R(A)
    if c_in_A:
        # d in R(A*)
        if d_in_At:
            # beta == 0
            if np.isclose(beta, 0):
                # c in R(A), d in R(A*), beta = 0
                return inverse6()

            # beta != 0 
            else: 
                # c in R(A), d in R(A*), beta != 0
                inv3 = inverse3()
                inv5 = inverse5()
                ok_("3 and 5 CLOSE?:",np.allclose(inv3, inv5))
                return(inv5)

        # d not in R(A*)
        else: 
            # beta == 0
            if np.isclose(beta, 0):
                # c in R(A), d not in R(A*), beta = 0
                return inverse2()

            # beta != 0 
            else: 
                # c in R(A), d not in R(A*), beta != 0
                return inverse3()


    # c not in R(A)
    else:
        # d in R(A*)
        if d_in_At:
            # beta == 0
            if np.isclose(beta, 0):
                # c not in R(A), d in R(A*), beta = 0
                return inverse4()

            # beta != 0 
            else: 
                # c not in R(A), d in R(A*), beta != 0
                return inverse5()

        # d not in R(A*)
        else:       
            # beta == 0
            if np.isclose(beta, 0):
                # c not in R(A), d not in R(A*), beta = 0
                return inverse1()

            # beta != 0 
            else: 
                # c not in R(A), d not in R(A*), beta != 0
                return inverse1()
    print("UHOH")


A = np.array([[1,1,1],[0,1,0],[1,1,0]])
c = np.array([1,0,0]).reshape(-1,1)
d = np.array([0.5,0.5,0]).reshape(-1,1)

Ainv = np.linalg.pinv(A)
Au = A + c @ d.T
Auinv = meyer_update(A, Ainv, c, d)
ok_(np.allclose(Au, Au@Auinv@Au))

A = np.array([[1,2,0,1],[0,1,-1,0],[0,0,1,-1]])
c = np.array([0,0,1]).reshape(-1,1)
d = np.array([0,0,-1,0]).reshape(-1,1)

Ainv = np.linalg.pinv(A)
Au = A + c @ d.T
Auinv = meyer_update(A, Ainv, c, d)
ok_(np.allclose(Au, Au@Auinv@Au))

