import numpy as np

'''
biconjugate gradient method

A : n x n positive defined matrix
b : vector of shape n 
x0: initial guess of solution
itmax: the maximum number of allowed iterations
'''

def biconjugate_gradient(A, b, x0 = np.array([1., 1., 1.]), eps = 1e-6, itmax = 100):
    x = x0.reshape(-1, 1); b = b.reshape(-1, 1); 
    r = b - np.dot(A, x0).reshape(-1, 1); p = r
    while True:
        rr = np.random.rand(len(r)).reshape(-1, 1)
        if np.dot(r.T, rr) != 0: break;
    pp = rr
    it = 0
    while it < itmax:
        alpha = (np.dot(r.T, rr)) / np.dot(np.dot(A, p).T, pp)
        x = x + alpha * p
        rh = r; rrh = rr
        r = r - alpha * np.dot(A, p)
        rr = rr - alpha * np.dot(A.T, pp)  
        if np.sqrt(np.dot(r.T, r)) < eps:
            print('sucessfully find solution')
            print('iter times: ', it)
            return x.flatten()
        beta = np.dot(r.T, rr) / np.dot(rh.T, rrh)
        p = r + beta * p
        pp = rr + beta * pp
        it += 1
    if it >= itmax:
        print('failed to find solution, becaise it > itmax')
        return

def bicg_test():
    A = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]], dtype = np.float64)
    b = np.array([1, 1, 1], dtype = np.float64)
    x0 = np.array([100, 100, 100], dtype = np.float64)
    res = biconjugate_gradient(A, b, x0)
    print('solution of Ax = b :', res)

if __name__ == '__main__':
    bicg_test()
