import numpy as np

'''
conjugate gradient method

A : n x n symmetric positive defined matrix
b : vector of shape n 
x0: initial guess of solution
itmax: the maximum number of allowed iterations
'''

def conjugate_gradient(A, b, x0 = np.array([1., 1., 1.]), eps = 1e-9, itmax = 100):
    x = x0.reshape(-1, 1); b = b.reshape(-1, 1)
    r = b - np.dot(A, x0).reshape(-1, 1) ; p = r
    it = 0
    while it < itmax:
        alpha = np.dot(r.T, r) / np.dot(np.dot(A, p).T, p)
        x = x + alpha * p
        rh = r
        r = r - alpha * np.dot(A, p)
        if np.sqrt(np.dot(r.T, r)) < eps:
            print('sucessfully find solution')
            print('iter times: ', it)
            return x.flatten()
        beta = np.dot(r.T, r) / np.dot(rh.T, rh)
        p = r + beta * p
        it += 1
    if it >= itmax:
        print('failed to find solution, becaise it > itmax')
        return 

def cg_test():
    A = np.array([[1, 0, 0], [0, 5, 0], [0, 0, 3]], dtype = np.float64)
    b = np.array([1, 2, 3], dtype = np.float64)
    x0 = np.array([10, 10, 10], dtype = np.float64)
    eps = 1e-9
    res = conjugate_gradient(A, b, x0, eps)
    print('solution of Ax = b: ', res)

if __name__ == '__main__':
    cg_test()
